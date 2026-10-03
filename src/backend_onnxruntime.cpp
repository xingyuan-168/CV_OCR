#include "backend_onnxruntime.h"
#include "yolo_preprocess.h"
#include "yolo_decode.h"
#include "yolo_diagnostics.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <onnxruntime_cxx_api.h>
#if defined(AIENGINE_EXPERIMENTAL_DML1)
#include <dml_provider_factory.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "image_view.h"
#include "ocr_reading_order.h"
#include "runtime_status.h"
#include "error.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#endif

#if defined(AIENGINE_EMBEDDED_ORT_RUNTIME)
#include "runtime_bundle_loader.h"
#endif

namespace ai {

namespace {

#ifndef AIENGINE_RUNTIME_FLAVOR_NAME
#define AIENGINE_RUNTIME_FLAVOR_NAME "core"
#endif

// 共享 ONNX Runtime 环境。ORT 要求 Env 生命周期长于所有 session。
Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ai_engine_dll");
    return env;
}

// 选择非空 C 字符串覆盖值；为空时使用配置 fallback。
std::string first_non_empty(const char* value, const std::string& fallback) {
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

// 将 "RGB"、"bgr" 等 ASCII 配置字符串转为小写。
std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

// 检查当前 ORT 构建/环境是否暴露指定执行 provider。
bool provider_available(const char* provider_name) {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(), provider_name) != providers.end();
}

std::vector<std::string> available_providers() {
    try {
        return Ort::GetAvailableProviders();
    } catch (...) {
        return {};
    }
}

std::string ort_version() {
#if defined(_WIN32)
    HMODULE module = GetModuleHandleW(L"onnxruntime.dll");
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    const auto get_api_base = module == nullptr
        ? nullptr
        : reinterpret_cast<GetApiBaseFn>(
            GetProcAddress(module, "OrtGetApiBase"));
    const OrtApiBase* base =
        get_api_base == nullptr ? nullptr : get_api_base();
#else
    const OrtApiBase* base = OrtGetApiBase();
#endif
    return base == nullptr || base->GetVersionString == nullptr
        ? std::string()
        : std::string(base->GetVersionString());
}

#if defined(_WIN32)
std::string wide_to_utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string output(static_cast<size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), output.data(), required, nullptr, nullptr);
    return output;
}

std::string loaded_ort_path() {
    HMODULE module = GetModuleHandleW(L"onnxruntime.dll");
    if (module == nullptr) return {};
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) return {};
    path.resize(length);
    return wide_to_utf8(path);
}
#else
std::string loaded_ort_path() {
    return {};
}
#endif

#if defined(_WIN32)
bool validate_directml_adapter(
    int32_t device_id,
    std::string* adapter_name,
    std::string* error) {
    if (device_id < 0) {
        if (error != nullptr) *error = "DirectML device_id must be non-negative";
        return false;
    }
    IDXGIFactory1* factory = nullptr;
    HRESULT result = CreateDXGIFactory1(
        __uuidof(IDXGIFactory1),
        reinterpret_cast<void**>(&factory));
    if (FAILED(result) || factory == nullptr) {
        if (error != nullptr) *error = "Cannot create DXGI factory for DirectML";
        return false;
    }
    IDXGIAdapter1* adapter = nullptr;
    result = factory->EnumAdapters1(
        static_cast<UINT>(device_id),
        &adapter);
    factory->Release();
    if (result == DXGI_ERROR_NOT_FOUND || adapter == nullptr) {
        if (error != nullptr) {
            *error = "DirectML adapter index " +
                std::to_string(device_id) + " is out of range";
        }
        return false;
    }

    DXGI_ADAPTER_DESC1 description{};
    result = adapter->GetDesc1(&description);
    if (FAILED(result)) {
        adapter->Release();
        if (error != nullptr) *error = "Cannot query DirectML adapter";
        return false;
    }
    if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        adapter->Release();
        if (error != nullptr) {
            *error = "DirectML adapter " + std::to_string(device_id) +
                " is a software adapter";
        }
        return false;
    }

    ID3D12Device* d3d12_device = nullptr;
    result = D3D12CreateDevice(
        adapter,
        D3D_FEATURE_LEVEL_11_0,
        __uuidof(ID3D12Device),
        reinterpret_cast<void**>(&d3d12_device));
    adapter->Release();
    if (FAILED(result) || d3d12_device == nullptr) {
        if (error != nullptr) {
            *error = "DirectX 12 is unavailable on adapter " +
                std::to_string(device_id);
        }
        return false;
    }
    d3d12_device->Release();
    if (adapter_name != nullptr) {
        *adapter_name = wide_to_utf8(description.Description);
    }
    return true;
}

#if defined(AIENGINE_EXPERIMENTAL_DML1)
bool append_directml1_provider(
    Ort::SessionOptions* options,
    int32_t device_id,
    std::string* error) {
    IDXGIFactory1* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device* d3d12_device = nullptr;
    ID3D12CommandQueue* command_queue = nullptr;
    IDMLDevice* dml_device = nullptr;
    const auto release_objects = [&]() {
        if (dml_device != nullptr) dml_device->Release();
        if (command_queue != nullptr) command_queue->Release();
        if (d3d12_device != nullptr) d3d12_device->Release();
        if (adapter != nullptr) adapter->Release();
        if (factory != nullptr) factory->Release();
    };

    HRESULT result = CreateDXGIFactory1(
        __uuidof(IDXGIFactory1),
        reinterpret_cast<void**>(&factory));
    if (FAILED(result) || factory == nullptr) {
        if (error != nullptr) {
            *error = "DML1 cannot create DXGI factory";
        }
        release_objects();
        return false;
    }
    result = factory->EnumAdapters1(
        static_cast<UINT>(device_id), &adapter);
    if (FAILED(result) || adapter == nullptr) {
        if (error != nullptr) {
            *error = "DML1 cannot open adapter " +
                std::to_string(device_id);
        }
        release_objects();
        return false;
    }
    result = D3D12CreateDevice(
        adapter,
        D3D_FEATURE_LEVEL_11_0,
        __uuidof(ID3D12Device),
        reinterpret_cast<void**>(&d3d12_device));
    if (FAILED(result) || d3d12_device == nullptr) {
        if (error != nullptr) {
            *error = "DML1 cannot create D3D12 device";
        }
        release_objects();
        return false;
    }
    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    result = d3d12_device->CreateCommandQueue(
        &queue_description,
        __uuidof(ID3D12CommandQueue),
        reinterpret_cast<void**>(&command_queue));
    if (FAILED(result) || command_queue == nullptr) {
        if (error != nullptr) {
            *error = "DML1 cannot create D3D12 compute queue";
        }
        release_objects();
        return false;
    }

#if defined(AIENGINE_EMBEDDED_ORT_RUNTIME)
    const std::filesystem::path directml_path =
        std::filesystem::u8path(
            ai_runtime::runtime_cache_path_utf8()) /
        L"DirectML.dll";
    HMODULE directml_module = GetModuleHandleW(L"DirectML.dll");
    if (directml_module == nullptr) {
        directml_module = LoadLibraryExW(
            directml_path.c_str(),
            nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
#else
    HMODULE directml_module = LoadLibraryW(L"DirectML.dll");
#endif
    if (directml_module == nullptr) {
        if (error != nullptr) {
            *error =
                "DML1 failed to load DirectML.dll: Windows error " +
                std::to_string(GetLastError());
        }
        release_objects();
        return false;
    }
    using DmlCreateDeviceFn = HRESULT(WINAPI*)(
        ID3D12Device*,
        DML_CREATE_DEVICE_FLAGS,
        REFIID,
        void**);
    const auto create_dml_device =
        reinterpret_cast<DmlCreateDeviceFn>(
            GetProcAddress(directml_module, "DMLCreateDevice"));
    if (create_dml_device == nullptr) {
        if (error != nullptr) {
            *error = "DML1 entry point DMLCreateDevice is unavailable";
        }
        release_objects();
        return false;
    }
    result = create_dml_device(
        d3d12_device,
        DML_CREATE_DEVICE_FLAG_NONE,
        __uuidof(IDMLDevice),
        reinterpret_cast<void**>(&dml_device));
    if (FAILED(result) || dml_device == nullptr) {
        if (error != nullptr) {
            *error = "DML1 cannot create DirectML device";
        }
        release_objects();
        return false;
    }

    const OrtDmlApi* dml_api = nullptr;
    OrtStatus* status = Ort::GetApi().GetExecutionProviderApi(
        "DML",
        ORT_API_VERSION,
        reinterpret_cast<const void**>(&dml_api));
    if (status != nullptr) {
        if (error != nullptr) {
            *error = Ort::GetApi().GetErrorMessage(status);
        }
        Ort::GetApi().ReleaseStatus(status);
        release_objects();
        return false;
    }
    if (dml_api == nullptr) {
        if (error != nullptr) {
            *error = "DML1 provider API is unavailable";
        }
        release_objects();
        return false;
    }
    status = dml_api->SessionOptionsAppendExecutionProvider_DML1(
        static_cast<OrtSessionOptions*>(*options),
        dml_device,
        command_queue);
    if (status != nullptr) {
        if (error != nullptr) {
            *error = Ort::GetApi().GetErrorMessage(status);
        }
        Ort::GetApi().ReleaseStatus(status);
        release_objects();
        return false;
    }
    release_objects();
    return true;
}
#endif

// DirectML provider 工厂通过已加载的官方 ORT 模块动态解析。
bool append_directml_provider(Ort::SessionOptions* options, int32_t device_id, std::string* error) {
#if defined(AIENGINE_EXPERIMENTAL_DML1)
    std::string dml1_error;
    if (append_directml1_provider(
            options, device_id, &dml1_error)) {
        return true;
    }
#endif
    using AppendDmlFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
    HMODULE module = GetModuleHandleA("onnxruntime.dll");
    if (module == nullptr) {
        if (error != nullptr) *error = "onnxruntime.dll is not loaded";
        return false;
    }
    const auto append = reinterpret_cast<AppendDmlFn>(GetProcAddress(module, "OrtSessionOptionsAppendExecutionProvider_DML"));
    if (append == nullptr) {
        if (error != nullptr) *error = "DirectML provider entry point is not available in this ONNX Runtime build";
        return false;
    }
    OrtStatus* status = append(static_cast<OrtSessionOptions*>(*options), std::max<int32_t>(0, device_id));
    if (status == nullptr) return true;
    if (error != nullptr) *error = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return false;
}
#else
bool validate_directml_adapter(int32_t, std::string*, std::string* error) {
    if (error != nullptr) *error = "DirectML is only available on Windows";
    return false;
}

bool append_directml_provider(Ort::SessionOptions*, int32_t, std::string* error) {
    if (error != nullptr) *error = "DirectML is only available on Windows";
    return false;
}
#endif

const char* requested_device_name(int32_t device) {
    switch (device) {
        case AI_DEVICE_AUTO: return "auto";
        case AI_DEVICE_DIRECTML: return "directml";
        case AI_DEVICE_CPU: return "cpu";
        default: return "invalid";
    }
}

void set_active_provider(
    int32_t requested,
    const char* active,
    bool degraded,
    std::string reason = {},
    int32_t device_id = 0,
    std::string adapter_name = {}) {
    RuntimeStatus status;
    status.requested = requested_device_name(requested);
    status.active = active;
    status.degraded = degraded;
    status.reason = std::move(reason);
    status.runtime_flavor = AIENGINE_RUNTIME_FLAVOR_NAME;
    status.ort_version = ort_version();
#if defined(AIENGINE_EMBEDDED_ORT_RUNTIME)
    status.ort_path = ai_runtime::loaded_ort_path_utf8();
#else
    status.ort_path = loaded_ort_path();
#endif
    status.available_providers = available_providers();
    status.mixed_cpu_fallback = std::strcmp(active, "directml") == 0;
    status.device_id = device_id;
    status.adapter_name = std::move(adapter_name);
    set_runtime_status(std::move(status));
}

// 配置统一的 CPU/DirectML/AUTO provider 选择。
bool configure_execution_provider(const Config& config, Ort::SessionOptions* options, std::string* error) {
    const int32_t runtime_device = config.get_int("runtime.device", AI_DEVICE_AUTO);
    const int32_t device_id = config.get_int("runtime.device_id", 0);
    if (runtime_device == AI_DEVICE_CPU) {
        set_active_provider(runtime_device, "cpu", false, {}, device_id);
        return true;
    }

    std::vector<std::string> failures;
    const auto try_directml = [&]() {
#if defined(AIENGINE_EMBEDDED_ORT_RUNTIME)
        std::string extraction_error;
        if (!ai_runtime::ensure_embedded_directml(&extraction_error)) {
            failures.emplace_back(
                extraction_error.empty()
                    ? "Failed to extract DirectML.dll"
                    : extraction_error);
            return false;
        }
#endif
        if (!provider_available("DmlExecutionProvider")) {
            failures.emplace_back("DmlExecutionProvider is unavailable");
            return false;
        }
        std::string adapter_name;
        std::string adapter_error;
        if (!validate_directml_adapter(
                device_id, &adapter_name, &adapter_error)) {
            failures.emplace_back(adapter_error);
            return false;
        }
        std::string provider_error;
        if (!append_directml_provider(options, device_id, &provider_error)) {
            failures.emplace_back(provider_error.empty() ? "failed to append DirectML provider" : provider_error);
            return false;
        }
        options->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options->DisableMemPattern();
        set_active_provider(
            runtime_device,
            "directml",
            false,
            {},
            device_id,
            std::move(adapter_name));
        return true;
    };

    if (runtime_device == AI_DEVICE_DIRECTML) {
        if (try_directml()) return true;
    } else if (runtime_device == AI_DEVICE_AUTO) {
        if (try_directml()) return true;
        const std::string reason = failures.empty()
            ? "AUTO fallback to CPU because DirectML initialization failed"
            : "AUTO fallback to CPU because DirectML initialization failed: " +
                failures.back();
        set_active_provider(runtime_device, "cpu", true, reason, device_id);
        return true;
    } else {
        if (error != nullptr) {
            *error = "Invalid runtime device " +
                std::to_string(runtime_device) +
                "; valid values are 0=AUTO, 1=DirectML, 2=CPU";
        }
        return false;
    }

    if (error != nullptr) {
        *error = failures.empty() ? "requested execution provider is unavailable" : failures.back();
    }
    return false;
}

// 构建路径加载和内存加载共用的 ORT session 选项。
bool make_session_options(const Config& config, Ort::SessionOptions* options, std::string* error, bool force_cpu = false) {
    options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(_WIN32)
    // Internal validation hook used by the release matrix. It is disabled in
    // normal operation and does not change the public ABI.
    wchar_t profile_prefix[32768]{};
    const DWORD profile_length = GetEnvironmentVariableW(
        L"CQ_AI_ORT_PROFILE_PREFIX",
        profile_prefix,
        static_cast<DWORD>(std::size(profile_prefix)));
    if (profile_length > 0 && profile_length < std::size(profile_prefix)) {
        options->EnableProfiling(profile_prefix);
    }
#endif
    int32_t thread_count = config.get_int("runtime.intra_op_threads", -1);
    if (thread_count < 0) {
        thread_count = config.get_int("runtime.thread_count", 1);
    }
    if (thread_count == 0) {
        const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
        const int32_t sessions = std::max<int32_t>(1, config.get_int("runtime.session_count", 1));
        const unsigned int per_session = std::max(1u, hw / static_cast<unsigned int>(sessions));
        thread_count = static_cast<int32_t>(std::min<unsigned int>(4u, per_session));
    }
    if (thread_count > 0) {
        options->SetIntraOpNumThreads(thread_count);
    }
    const int32_t inter_threads = config.get_int("runtime.inter_op_threads", 1);
    if (inter_threads > 0) {
        options->SetInterOpNumThreads(inter_threads);
    }
    if (force_cpu) {
        return true;
    }
    return configure_execution_provider(config, options, error);
}

// AUTO 模式下，session 创建失败会用仅 CPU 选项重试。
bool should_retry_cpu(const Config& config) {
    return config.get_int("runtime.device", AI_DEVICE_AUTO) == AI_DEVICE_AUTO;
}

#if defined(_WIN32)
// 将 UTF-8 模型路径转成宽字符路径，确保 Windows 中文目录可用。
std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) {
        return std::wstring();
    }

    const int required = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (required <= 0) {
        return std::wstring(value.begin(), value.end());
    }

    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, &output[0], required);
    if (!output.empty() && output.back() == L'\0') {
        output.pop_back();
    }
    return output;
}
#endif

// 从模型路径创建 ORT session；AUTO+DirectML 失败时重试 CPU。
std::unique_ptr<Ort::Session> create_session(const std::string& model_path, const Config& config, std::string* error) {
    try {
        Ort::SessionOptions options;
        if (!make_session_options(config, &options, error)) {
            return nullptr;
        }

#if defined(_WIN32)
        const std::wstring wide_path = utf8_to_wide(model_path);
        return std::make_unique<Ort::Session>(ort_env(), wide_path.c_str(), options);
#else
        return std::make_unique<Ort::Session>(ort_env(), model_path.c_str(), options);
#endif
    } catch (const Ort::Exception& ex) {
        if (should_retry_cpu(config)) {
            try {
                Ort::SessionOptions cpu_options;
                if (make_session_options(config, &cpu_options, nullptr, true)) {
#if defined(_WIN32)
                    const std::wstring wide_path = utf8_to_wide(model_path);
                    set_active_provider(
                        AI_DEVICE_AUTO,
                        "cpu",
                        true,
                        std::string("AUTO fallback to CPU because DirectML session creation failed: ") +
                            ex.what(),
                        config.get_int("runtime.device_id", 0));
                    return std::make_unique<Ort::Session>(ort_env(), wide_path.c_str(), cpu_options);
#else
                    set_active_provider(
                        AI_DEVICE_AUTO,
                        "cpu",
                        true,
                        std::string("AUTO fallback to CPU because DirectML session creation failed: ") +
                            ex.what(),
                        config.get_int("runtime.device_id", 0));
                    return std::make_unique<Ort::Session>(ort_env(), model_path.c_str(), cpu_options);
#endif
                }
            } catch (...) {
            }
        }
        if (error != nullptr) {
            *error = ex.what();
        }
        return nullptr;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = ex.what();
        }
        return nullptr;
    }
}

// 从内存 ONNX 字节创建 ORT session；AUTO+DirectML 失败时重试 CPU。
std::unique_ptr<Ort::Session> create_session_from_memory(const void* model_data, size_t model_size, const Config& config, std::string* error) {
    if (model_data == nullptr || model_size == 0) {
        if (error != nullptr) {
            *error = "Model memory is empty";
        }
        return nullptr;
    }

    try {
        Ort::SessionOptions options;
        if (!make_session_options(config, &options, error)) {
            return nullptr;
        }
        return std::make_unique<Ort::Session>(ort_env(), model_data, model_size, options);
    } catch (const Ort::Exception& ex) {
        if (should_retry_cpu(config)) {
            try {
                Ort::SessionOptions cpu_options;
                if (make_session_options(config, &cpu_options, nullptr, true)) {
                    set_active_provider(
                        AI_DEVICE_AUTO,
                        "cpu",
                        true,
                        std::string("AUTO fallback to CPU because DirectML session creation failed: ") +
                            ex.what(),
                        config.get_int("runtime.device_id", 0));
                    return std::make_unique<Ort::Session>(ort_env(), model_data, model_size, cpu_options);
                }
            } catch (...) {
            }
        }
        if (error != nullptr) {
            *error = ex.what();
        }
        return nullptr;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = ex.what();
        }
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// YOLO 前处理/后处理辅助函数
// ---------------------------------------------------------------------------

// PP-OCR 检测模型使用独立的图像预处理约定：BGR 通道、ImageNet mean/std，
// 并把原图直接缩放到 32 对齐的动态输入尺寸。检测框映射使用独立的 X/Y 比例。
void ocr_detection_to_chw_float(
    const AIImage& image,
    int target_w,
    int target_h,
    std::vector<float>& output) {
    const int channels = channels_for_format(image.format);
    const bool source_bgr = image.format == AI_IMAGE_BGR24 || image.format == AI_IMAGE_BGRA32;
    const size_t plane_size = static_cast<size_t>(target_w) * target_h;
    output.resize(plane_size * 3);

    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    const float scale_x = static_cast<float>(image.width) / target_w;
    const float scale_y = static_cast<float>(image.height) / target_h;

    for (int dy = 0; dy < target_h; ++dy) {
        const float source_y = std::max(0.0f, (dy + 0.5f) * scale_y - 0.5f);
        const int y0 = std::min(static_cast<int>(source_y), image.height - 1);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float fy = source_y - y0;
        const uint8_t* row0 = image_row_ptr(image, y0);
        const uint8_t* row1 = image_row_ptr(image, y1);
        for (int dx = 0; dx < target_w; ++dx) {
            const float source_x = std::max(0.0f, (dx + 0.5f) * scale_x - 0.5f);
            const int x0 = std::min(static_cast<int>(source_x), image.width - 1);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float fx = source_x - x0;
            const size_t dst_index = static_cast<size_t>(dy) * target_w + dx;

            for (int c = 0; c < 3; ++c) {
                const int source_channel = channels == 1 ? 0 : (source_bgr ? c : 2 - c);
                const float top = row0[x0 * channels + source_channel] * (1.0f - fx) +
                    row0[x1 * channels + source_channel] * fx;
                const float bottom = row1[x0 * channels + source_channel] * (1.0f - fx) +
                    row1[x1 * channels + source_channel] * fx;
                const float value = (top * (1.0f - fy) + bottom * fy) / 255.0f;
                output[static_cast<size_t>(c) * plane_size + dst_index] = (value - mean[c]) / stddev[c];
            }
        }
    }
}

// 计算两个原图坐标框的交并比。
// 将 ORT 节点名复制到自持有字符串，确保 Run() 使用稳定的 c_str() 指针。
void cache_node_names(Ort::Session& session, std::vector<std::string>& input_names, std::vector<std::string>& output_names) {
    Ort::AllocatorWithDefaultOptions allocator;
    input_names.clear();
    output_names.clear();

    const size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names.emplace_back(name.get());
    }

    const size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(name.get());
    }
}

// ---------------------------------------------------------------------------
// OCR 前处理/后处理辅助函数
// ---------------------------------------------------------------------------

// OCR 文本区域候选框，可处于检测概率图坐标或原图坐标。
using TextBox = OcrReadingOrderBox;

// 对二值概率图做简单连通域分析。eight_connected 仅改变相邻像素的
// 归并方式，不会重复运行检测模型。
std::vector<TextBox> find_text_boxes(
    const float* prob_map,
    int map_w,
    int map_h,
    float threshold,
    int min_area,
    float box_score_threshold,
    float unclip_ratio,
    bool eight_connected = false) {
    // 第 1 步：二值化。
    std::vector<uint8_t> binary(static_cast<size_t>(map_w) * map_h, 0);
    for (int i = 0; i < map_w * map_h; ++i) {
        if (prob_map[i] >= threshold) {
            binary[i] = 1;
        }
    }

    // 第 2 步：简单并查集标记。
    std::vector<int> labels(static_cast<size_t>(map_w) * map_h, 0);
    std::vector<int> parent;
    parent.push_back(0); // label 0 = background
    int next_label = 1;

    // 带路径压缩的根节点查找。
    auto find_root = [&](int lbl) -> int {
        while (parent[lbl] != lbl) {
            parent[lbl] = parent[parent[lbl]];
            lbl = parent[lbl];
        }
        return lbl;
    };

    auto unite = [&](int a, int b) {
        int ra = find_root(a);
        int rb = find_root(b);
        if (ra != rb) {
            parent[rb] = ra;
        }
    };

    // 第一遍扫描。
    for (int y = 0; y < map_h; ++y) {
        for (int x = 0; x < map_w; ++x) {
            const int idx = y * map_w + x;
            if (binary[idx] == 0) continue;

            std::array<int, 4> neighbors{};
            size_t neighbor_count = 0;
            const auto add_neighbor = [&](int label) {
                if (label == 0) return;
                for (size_t i = 0; i < neighbor_count; ++i) {
                    if (neighbors[i] == label) return;
                }
                neighbors[neighbor_count++] = label;
            };
            add_neighbor((y > 0) ? labels[(y - 1) * map_w + x] : 0);
            add_neighbor((x > 0) ? labels[y * map_w + (x - 1)] : 0);
            if (eight_connected) {
                add_neighbor((y > 0 && x > 0)
                    ? labels[(y - 1) * map_w + x - 1] : 0);
                add_neighbor((y > 0 && x + 1 < map_w)
                    ? labels[(y - 1) * map_w + x + 1] : 0);
            }

            if (neighbor_count == 0) {
                parent.push_back(next_label);
                labels[idx] = next_label++;
            } else {
                labels[idx] = neighbors[0];
                for (size_t i = 1; i < neighbor_count; ++i) {
                    unite(neighbors[0], neighbors[i]);
                }
            }
        }
    }

    // 第二遍扫描：压平标签。
    for (int i = 0; i < map_w * map_h; ++i) {
        if (labels[i] > 0) {
            labels[i] = find_root(labels[i]);
        }
    }

    // 第 3 步：按标签计算边界框。
    struct BBox {
        int min_x, min_y, max_x, max_y;
        int area;
        float score_sum;
    };
    // 标签是连续编号，因此直接用 vector 做简单存储。
    std::vector<BBox> label_boxes(next_label, BBox{map_w, map_h, 0, 0, 0, 0.0f});

    for (int y = 0; y < map_h; ++y) {
        for (int x = 0; x < map_w; ++x) {
            const int lbl = labels[y * map_w + x];
            if (lbl == 0) continue;
            auto& bb = label_boxes[lbl];
            bb.min_x = std::min(bb.min_x, x);
            bb.min_y = std::min(bb.min_y, y);
            bb.max_x = std::max(bb.max_x, x);
            bb.max_y = std::max(bb.max_y, y);
            bb.area++;
            bb.score_sum += prob_map[y * map_w + x];
        }
    }

    // DB 概率图中的句号、冒号等小标点可能只有 1-2 个像素，单独看会
    // 低于 min_area/box_score，但它们属于紧邻的同一文本行。先把这类
    // 垂直对齐且水平相邻的小区域并入已经可靠的主区域，再执行过滤。
    std::vector<bool> primary(static_cast<size_t>(next_label), false);
    for (int lbl = 1; lbl < next_label; ++lbl) {
        const auto& bb = label_boxes[lbl];
        primary[static_cast<size_t>(lbl)] = bb.area >= min_area &&
            bb.max_x >= bb.min_x &&
            bb.score_sum / static_cast<float>(std::max(1, bb.area)) >= box_score_threshold;
    }
    for (int lbl = 1; lbl < next_label; ++lbl) {
        if (primary[static_cast<size_t>(lbl)]) continue;
        const auto& small_box = label_boxes[lbl];
        if (small_box.area <= 0 || small_box.max_x < small_box.min_x) continue;
        int best_primary = 0;
        int best_gap = std::numeric_limits<int>::max();
        for (int candidate = 1; candidate < next_label; ++candidate) {
            if (!primary[static_cast<size_t>(candidate)]) continue;
            const auto& main = label_boxes[candidate];
            const int main_h = main.max_y - main.min_y + 1;
            const int small_h = small_box.max_y - small_box.min_y + 1;
            const int vertical_gap = std::max(0, std::max(main.min_y, small_box.min_y) -
                std::min(main.max_y, small_box.max_y) - 1);
            if (vertical_gap > std::max(main_h, small_h) / 2) continue;
            const int horizontal_gap = small_box.min_x > main.max_x
                ? small_box.min_x - main.max_x - 1
                : (main.min_x > small_box.max_x ? main.min_x - small_box.max_x - 1 : 0);
            if (horizontal_gap > std::max(4, main_h)) continue;
            if (horizontal_gap < best_gap) {
                best_gap = horizontal_gap;
                best_primary = candidate;
            }
        }
        if (best_primary != 0) {
            auto& main = label_boxes[best_primary];
            main.min_x = std::min(main.min_x, small_box.min_x);
            main.min_y = std::min(main.min_y, small_box.min_y);
            main.max_x = std::max(main.max_x, small_box.max_x);
            main.max_y = std::max(main.max_y, small_box.max_y);
            main.area += small_box.area;
            main.score_sum += small_box.score_sum;
        }
    }

    // 第 4 步：收集结果并过滤过小区域。
    std::vector<TextBox> result;
    for (int lbl = 1; lbl < next_label; ++lbl) {
        const auto& bb = label_boxes[lbl];
        if (!primary[static_cast<size_t>(lbl)] || bb.area < min_area) continue;
        if (bb.max_x < bb.min_x) continue; // invalid
        if (bb.score_sum / static_cast<float>(bb.area) < box_score_threshold) continue;
        const int bw = bb.max_x - bb.min_x + 1;
        const int bh = bb.max_y - bb.min_y + 1;
        // Paddle DB 的 unclip 距离是 polygon_area * ratio / polygon_perimeter，
        // 不是简单把宽高乘以 ratio。当前候选是轴对齐连通域，因此用其
        // 外接矩形面积和周长计算等价的统一外扩距离。这个规则对很长、
        // 很薄的小字号文本尤其重要，否则上下笔画会在送入识别模型前丢失。
        const float area = static_cast<float>(bw) * bh;
        const float perimeter = 2.0f * (bw + bh);
        const int unclip_distance = perimeter > 0.0f
            ? static_cast<int>(std::ceil(area * std::max(0.0f, unclip_ratio) / perimeter))
            : 0;
        const int extra_w = unclip_distance;
        const int extra_h = unclip_distance;
        const int x = std::max(0, bb.min_x - extra_w);
        const int y = std::max(0, bb.min_y - extra_h);
        const int right = std::min(map_w, bb.max_x + extra_w + 1);
        const int bottom = std::min(map_h, bb.max_y + extra_h + 1);
        result.push_back(TextBox{x, y, std::max(1, right - x), std::max(1, bottom - y)});
    }

    return result;
}

// 贪心 CTC 解码结果：UTF-8 文本和非 blank 平均置信度。
struct CtcResult {
    std::string text;
    float confidence;
};

struct OcrRecRegion {
    TextBox box;
    int crop_x = 0;
    int crop_y = 0;
    int crop_w = 0;
    int crop_h = 0;
    int target_w = 0;
    int tensor_w = 0;
    size_t output_index = 0;
};

struct OcrDecodedRegion {
    CtcResult ctc;
    bool valid = false;
};

void replace_all_utf8(std::string* value, const char* from, const char* to) {
    if (value == nullptr || from == nullptr || from[0] == '\0' || to == nullptr) return;
    const size_t from_size = std::strlen(from);
    const size_t to_size = std::strlen(to);
    size_t position = 0;
    while ((position = value->find(from, position)) != std::string::npos) {
        value->replace(position, from_size, to);
        position += to_size;
    }
}

// OCR 光栅无法可靠区分部分全角/半角标点。易语言业务文本统一使用
// Windows 常见的 ASCII 冒号、感叹号和左方括号，因此在所有 OCR
// 输出与文本查找之前做通用码位规范化；不包含任何业务词句替换。
void normalize_ocr_punctuation(std::string* text) {
    replace_all_utf8(text, u8"：", ":");
    replace_all_utf8(text, u8"！", "!");
    replace_all_utf8(text, u8"【", "[");
}

// CTC 贪心解码：每个时间步取 argmax，合并重复项，并移除 blank（索引 0）。
CtcResult ctc_greedy_decode(const float* logits, int seq_len, int num_classes, const std::vector<std::string>& charset) {
    CtcResult result;
    result.confidence = 0.0f;

    if (seq_len <= 0 || num_classes <= 0) {
        return result;
    }

    std::vector<int> best_path;
    std::vector<float> best_probs;
    best_path.reserve(seq_len);
    best_probs.reserve(seq_len);

    for (int t = 0; t < seq_len; ++t) {
        const float* row = logits + t * num_classes;
        int best_idx = 0;
        float best_val = row[0];
        float min_val = row[0];
        float sum_val = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > best_val) {
                best_val = row[c];
                best_idx = c;
            }
            min_val = std::min(min_val, row[c]);
            sum_val += row[c];
        }

        float prob = best_val;
        const bool already_probabilities = min_val >= -1.0e-6f && best_val <= 1.0f + 1.0e-6f && sum_val > 0.95f && sum_val < 1.05f;
        if (!already_probabilities) {
            // 模型返回 logits 时，为 best_idx 计算 softmax 概率。
            const float max_val = best_val;
            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(row[c] - max_val);
            }
            prob = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        }

        best_path.push_back(best_idx);
        best_probs.push_back(prob);
    }

    // 合并重复索引并移除 blank（索引 0）。
    int prev_idx = -1;
    float conf_sum = 0.0f;
    int conf_count = 0;

    for (int t = 0; t < seq_len; ++t) {
        const int idx = best_path[t];
        if (idx == prev_idx) {
            continue; // skip repeated
        }
        prev_idx = idx;
        if (idx == 0) {
            continue; // skip blank
        }
        // 将索引映射为字符。按当前存储约定 charset_[0] 是 blank 占位，
        // 实际字符从 charset_[1] 开始。
        if (idx < static_cast<int>(charset.size())) {
            result.text += charset[idx];
        }
        conf_sum += best_probs[t];
        conf_count++;
    }

    result.confidence = (conf_count > 0) ? (conf_sum / conf_count) : 0.0f;
    return result;
}

double log_add(double left, double right) {
    if (!std::isfinite(left)) return right;
    if (!std::isfinite(right)) return left;
    const double high = std::max(left, right);
    return high + std::log(std::exp(left - high) + std::exp(right - high));
}

// Standard CTC prefix beam search. Blank and non-blank probabilities are kept
// separately so repeated symbols are only emitted after an intervening blank.
// Greedy remains the production default; beam is available to accuracy configs.
CtcResult ctc_beam_decode(const float* logits, int seq_len, int num_classes, const std::vector<std::string>& charset, int beam_width) {
    struct Probability {
        double blank = -std::numeric_limits<double>::infinity();
        double non_blank = -std::numeric_limits<double>::infinity();
    };
    std::map<std::vector<int>, Probability> beams;
    beams[{}].blank = 0.0;
    const int top_k = std::min(num_classes, std::max(beam_width * 2, 8));
    for (int t = 0; t < seq_len; ++t) {
        const float* row = logits + t * num_classes;
        std::vector<double> log_probs(static_cast<size_t>(num_classes));
        float max_val = row[0];
        double sum_exp = 0.0;
        for (int c = 1; c < num_classes; ++c) max_val = std::max(max_val, row[c]);
        for (int c = 0; c < num_classes; ++c) sum_exp += std::exp(static_cast<double>(row[c] - max_val));
        if (sum_exp <= 0.0) continue;
        const double log_sum_exp = std::log(sum_exp) + max_val;
        for (int c = 0; c < num_classes; ++c) log_probs[static_cast<size_t>(c)] = row[c] - log_sum_exp;

        std::vector<int> top_indices(num_classes);
        std::iota(top_indices.begin(), top_indices.end(), 0);
        std::partial_sort(top_indices.begin(), top_indices.begin() + top_k, top_indices.end(), [&log_probs](int a, int b) {
            return log_probs[static_cast<size_t>(a)] > log_probs[static_cast<size_t>(b)];
        });
        top_indices.resize(static_cast<size_t>(top_k));
        if (std::find(top_indices.begin(), top_indices.end(), 0) == top_indices.end()) top_indices.back() = 0;

        std::map<std::vector<int>, Probability> next;
        for (const auto& beam : beams) {
            const std::vector<int>& prefix = beam.first;
            const Probability& probability = beam.second;
            const double total = log_add(probability.blank, probability.non_blank);
            Probability& same = next[prefix];
            same.blank = log_add(same.blank, total + log_probs[0]);

            for (const int symbol : top_indices) {
                if (symbol == 0 || symbol >= static_cast<int>(charset.size())) continue;
                const double symbol_log_prob = log_probs[static_cast<size_t>(symbol)];
                if (!prefix.empty() && prefix.back() == symbol) {
                    same.non_blank = log_add(same.non_blank, probability.non_blank + symbol_log_prob);
                    std::vector<int> extended = prefix;
                    extended.push_back(symbol);
                    Probability& destination = next[extended];
                    destination.non_blank = log_add(destination.non_blank, probability.blank + symbol_log_prob);
                } else {
                    std::vector<int> extended = prefix;
                    extended.push_back(symbol);
                    Probability& destination = next[extended];
                    destination.non_blank = log_add(destination.non_blank, total + symbol_log_prob);
                }
            }
        }

        std::vector<std::pair<std::vector<int>, Probability>> ranked(next.begin(), next.end());
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            const double left_score = log_add(left.second.blank, left.second.non_blank);
            const double right_score = log_add(right.second.blank, right.second.non_blank);
            if (left_score != right_score) return left_score > right_score;
            return left.first < right.first;
        });
        if (static_cast<int>(ranked.size()) > beam_width) ranked.resize(static_cast<size_t>(beam_width));
        beams.clear();
        for (auto& candidate : ranked) beams.emplace(std::move(candidate));
    }

    CtcResult result;
    if (!beams.empty()) {
        const auto best = std::max_element(beams.begin(), beams.end(), [](const auto& left, const auto& right) {
            return log_add(left.second.blank, left.second.non_blank) <
                log_add(right.second.blank, right.second.non_blank);
        });
        for (const int symbol : best->first) result.text += charset[static_cast<size_t>(symbol)];
        // Keep confidence on the established non-blank average scale so decoder
        // selection does not silently change min_confidence behavior.
        result.confidence = ctc_greedy_decode(logits, seq_len, num_classes, charset).confidence;
    }
    return result;
}

// 加载字符集文件：每行一个 UTF-8 字符。
// 返回的 vector 中索引 0 为 ""（blank），索引 1.. 为实际字符。
bool load_charset_file(const std::string& path, std::vector<std::string>& charset) {
    charset.clear();
    charset.push_back(""); // index 0 = CTC blank

    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        // 去掉可能存在的行尾 \r。
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        charset.push_back(line);
    }

    return charset.size() > 1;
}

// 从内联 UTF-8 文本读取每行标签，供内置资源模式避免外部 classes.txt。
void load_lines_from_inline_text(const std::string& text, std::vector<std::string>& lines, bool add_blank_prefix) {
    lines.clear();
    if (add_blank_prefix) {
        lines.push_back("");
    }

    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const bool is_trailing_empty_line = line.empty() && end == std::string::npos && start == text.size();
        if (!line.empty() || (add_blank_prefix && !is_trailing_empty_line)) {
            lines.push_back(line);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// OnnxRuntimeYoloBackend
// ---------------------------------------------------------------------------

namespace {

// 基于 ONNX Runtime session 的 YOLO 后端。
// 支持从文件系统路径或内存模型缓冲区加载。
class OnnxRuntimeYoloBackend final : public YoloBackend {
public:
    // 保存初始配置；session 由 load_model() 创建。
    explicit OnnxRuntimeYoloBackend(Config config) : config_(std::move(config)) {}

    // 从磁盘加载 YOLO ONNX 模型，并刷新所有模型相关设置。
    int32_t load_model(const char* model_path, const Config& config, std::string* error) override {
        config_ = config;
        model_path_ = first_non_empty(model_path, config_.get_string("yolo.model_path", ""));
        session_.reset();

        if (model_path_.empty()) {
            loaded_without_model_ = true;
            return AI_OK;
        }

        session_ = create_session(model_path_, config_, error);
        loaded_without_model_ = false;
        if (!session_) {
            return AI_ERR_RUNTIME;
        }

        if (!read_yolo_config(error)) {
            session_.reset();
            return AI_ERR_INVALID_ARGUMENT;
        }
        cache_node_names(*session_, input_names_cache_, output_names_cache_);
        refresh_node_name_ptrs();
        return AI_OK;
    }

    // 从模型上下文持有的稳定字节加载 YOLO ONNX 模型；上下文只保存一份模型数据供全部 Session 创建。
    int32_t load_model_from_memory(const void* model_data, size_t model_size, const Config& config, std::string* error) override {
        if (model_data == nullptr || model_size == 0) {
            return AI_ERR_INVALID_ARGUMENT;
        }

        config_ = config;
        model_path_.clear();
        session_.reset();

        session_ = create_session_from_memory(model_data, model_size, config_, error);
        loaded_without_model_ = false;
        if (!session_) {
            return AI_ERR_RUNTIME;
        }

        if (!read_yolo_config(error)) {
            session_.reset();
            return AI_ERR_INVALID_ARGUMENT;
        }
        cache_node_names(*session_, input_names_cache_, output_names_cache_);
        refresh_node_name_ptrs();
        return AI_OK;
    }

    // 执行前处理、ONNX 推理、YOLO 后处理和 NMS。
    int32_t detect(
        const AIImage& image,
        float conf_threshold,
        float nms_threshold,
        std::vector<AIDetectBox>* output) override {
        if (output == nullptr) return AI_ERR_INVALID_ARGUMENT;
        output->clear();
        if (!session_) {
            return loaded_without_model_ ? AI_ERR_BACKEND_NOT_CONFIGURED : AI_ERR_RUNTIME;
        }

        const auto preprocess_start = YoloClock::now();
        if (!input_value_) {
            input_tensor_data_.resize(static_cast<size_t>(3) * input_width_ * input_height_);
            const std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
            const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            input_value_ = Ort::Value::CreateTensor<float>(memory, input_tensor_data_.data(),
                input_tensor_data_.size(), input_shape.data(), input_shape.size());
            if (output_name_ptrs_.size() == 1) {
                const auto type = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
                const auto shape = type.GetShape();
                if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    shape.size() == 3 && std::all_of(shape.begin(), shape.end(), [](int64_t n) { return n > 0; })) {
                    output_tensor_data_.resize(type.GetElementCount());
                    output_value_ = Ort::Value::CreateTensor<float>(memory, output_tensor_data_.data(),
                        output_tensor_data_.size(), shape.data(), shape.size());
                }
            }
        }
        const auto lbox = yolo_preprocess(image, input_width_, input_height_, input_tensor_data_.data(), columns_);
        yolo_timing.preprocess_us = elapsed_us(preprocess_start);
        const auto run_start = YoloClock::now();
        std::vector<Ort::Value> output_values;
        try {
            if (output_value_) {
                session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), &input_value_, 1,
                    output_name_ptrs_.data(), &output_value_, 1);
            } else {
                output_values = session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(),
                    &input_value_, input_name_ptrs_.size(), output_name_ptrs_.data(), output_name_ptrs_.size());
            }
        } catch (const Ort::Exception& e) {
            set_last_error(e.what());
            return AI_ERR_RUNTIME;
        }
        yolo_timing.run_us = elapsed_us(run_start);
        const auto postprocess_start = YoloClock::now();
        if (!output_value_ && output_values.empty()) {
            return 0;
        }

        // 5. 后处理：输出形状通常为 [1, 84, 8400]（YOLOv8 格式）。
        const auto& out_tensor = output_value_ ? output_value_ : output_values[0];
        const auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
        const auto shape = type_info.GetShape();

        // 期望形状为 [1, num_attrs, num_candidates]，其中 num_attrs = 4 + num_classes。
        if (shape.size() != 3 || shape[0] != 1 ||
            type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return AI_ERR_RUNTIME;
        }

        const int32_t result_count = yolo_decode(out_tensor.GetTensorData<float>(), shape[1], shape[2],
            image, lbox, conf_threshold, nms_threshold, labels_, candidates_, output);
        yolo_timing.postprocess_us = elapsed_us(postprocess_start);
        return result_count;
    }

    // 释放当前 ORT session 和所有复制的内存模型数据。
    int32_t release_model() override {
        input_value_ = Ort::Value{nullptr};
        output_value_ = Ort::Value{nullptr};
        session_.reset();
        loaded_without_model_ = false;
        return AI_OK;
    }

    int32_t input_width() const override { return input_width_; }
    int32_t input_height() const override { return input_height_; }

private:
    void refresh_node_name_ptrs() {
        input_name_ptrs_.clear();
        input_name_ptrs_.reserve(input_names_cache_.size());
        for (const auto& name : input_names_cache_) {
            input_name_ptrs_.push_back(name.c_str());
        }
        output_name_ptrs_.clear();
        output_name_ptrs_.reserve(output_names_cache_.size());
        for (const auto& name : output_names_cache_) {
            output_name_ptrs_.push_back(name.c_str());
        }
    }

    // 从当前配置读取输入尺寸、阈值和标签元数据。
    bool read_yolo_config(std::string* error) {
        const std::vector<int64_t> shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] != 3) {
            if (error != nullptr) *error = "YOLO input must be a four-dimensional NCHW tensor with three channels";
            return false;
        }

        input_width_ = config_.get_int("yolo.input_width", 0);
        input_height_ = config_.get_int("yolo.input_height", 0);
        if (input_width_ == 0) {
            if (shape[3] <= 0) {
                if (error != nullptr) *error = "dynamic YOLO input width must be configured explicitly";
                return false;
            }
            input_width_ = static_cast<int32_t>(shape[3]);
        } else if (input_width_ < 0 || (shape[3] > 0 && shape[3] != input_width_)) {
            if (error != nullptr) *error = "configured YOLO input width does not match the static ONNX input shape";
            return false;
        }
        if (input_height_ == 0) {
            if (shape[2] <= 0) {
                if (error != nullptr) *error = "dynamic YOLO input height must be configured explicitly";
                return false;
            }
            input_height_ = static_cast<int32_t>(shape[2]);
        } else if (input_height_ < 0 || (shape[2] > 0 && shape[2] != input_height_)) {
            if (error != nullptr) *error = "configured YOLO input height does not match the static ONNX input shape";
            return false;
        }
        if (input_width_ <= 0 || input_height_ <= 0) {
            if (error != nullptr) *error = "YOLO input shape is dynamic; specify input_size as 320 or 640";
            return false;
        }
        if (input_width_ != input_height_) {
            if (error != nullptr) *error = "YOLO static input height and width must be equal";
            return false;
        }
        load_yolo_labels();
        return true;
    }

    // 加载可选类别标签，每行一个 UTF-8 标签。
    void load_yolo_labels() {
        labels_.clear();
        const std::string labels_inline = config_.get_string("yolo.labels_inline", "");
        if (!labels_inline.empty()) {
            load_lines_from_inline_text(labels_inline, labels_, false);
            return;
        }

        const std::string labels_path = config_.get_string("yolo.labels_path", "");
        if (labels_path.empty()) {
            return;
        }

        std::ifstream input(std::filesystem::u8path(labels_path), std::ios::in | std::ios::binary);
        if (!input.is_open()) {
            return;
        }

        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                labels_.push_back(line);
            }
        }
    }

    // 将类别标签复制到固定大小的 C ABI 输出字段。
    void copy_yolo_label(int class_id, char* output, int output_size) const {
        if (output == nullptr || output_size <= 0) {
            return;
        }
        output[0] = '\0';

        if (class_id < 0 || class_id >= static_cast<int>(labels_.size())) {
            return;
        }

        const std::string& label = labels_[static_cast<size_t>(class_id)];
        const size_t copy_len = std::min(label.size(), static_cast<size_t>(output_size - 1));
        std::memcpy(output, label.data(), copy_len);
        output[copy_len] = '\0';
    }

    Config config_;
    std::string model_path_;
    std::unique_ptr<Ort::Session> session_;
    bool loaded_without_model_ = false;

    std::vector<std::string> input_names_cache_;
    std::vector<std::string> output_names_cache_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<std::string> labels_;
    Ort::Value input_value_{nullptr}, output_value_{nullptr};
    std::vector<float> output_tensor_data_;
    std::vector<int> columns_;
    std::vector<float> input_tensor_data_;
    std::vector<RawDetection> candidates_;
    int input_width_ = 0;
    int input_height_ = 0;
};

// 基于 ONNX Runtime 的 PP-OCR 风格后端。
// 检测 session 可选；真实 OCR 必须有识别 session。
class OnnxRuntimeOcrBackend final : public OcrBackend {
public:
    // 保存初始配置；模型 session 由 load_models() 创建。
    explicit OnnxRuntimeOcrBackend(Config config) : config_(std::move(config)) {}

    // 从磁盘加载可选检测模型和必需识别模型。
    int32_t load_models(const char* det_model_path, const char* rec_model_path, const Config& config, std::string* error) override {
        config_ = config;
        const std::string det_path = first_non_empty(det_model_path, config_.get_string("ocr.det_model_path", ""));
        const std::string rec_path = first_non_empty(rec_model_path, config_.get_string("ocr.rec_model_path", ""));
        release_models();

        if (rec_path.empty()) {
            loaded_without_model_ = true;
            return AI_OK;
        }

        if (!det_path.empty()) {
            det_session_ = create_session(det_path, config_, error);
            if (!det_session_) {
                return AI_ERR_RUNTIME;
            }
        }

        rec_session_ = create_session(rec_path, config_, error);
        loaded_without_model_ = false;
        if (!rec_session_) {
            return AI_ERR_RUNTIME;
        }

        read_ocr_config();
        if (det_session_) {
            cache_node_names(*det_session_, det_input_names_, det_output_names_);
        }
        cache_node_names(*rec_session_, rec_input_names_, rec_output_names_);
        return AI_OK;
    }

    // 从内存缓冲区加载可选检测模型和必需识别模型。
    // 会复制缓冲区，使 ORT session 不依赖调用方内存生命周期。
    int32_t load_models_from_memory(
        const void* det_model_data,
        size_t det_model_size,
        const void* rec_model_data,
        size_t rec_model_size,
        const Config& config,
        std::string* error) override {
        if (rec_model_data == nullptr || rec_model_size == 0 || (det_model_data == nullptr && det_model_size > 0)) {
            return AI_ERR_INVALID_ARGUMENT;
        }

        config_ = config;
        release_models();

        if (det_model_data != nullptr && det_model_size > 0) {
            det_model_bytes_.assign(
                static_cast<const uint8_t*>(det_model_data),
                static_cast<const uint8_t*>(det_model_data) + det_model_size);
            det_session_ = create_session_from_memory(det_model_bytes_.data(), det_model_bytes_.size(), config_, error);
            if (!det_session_) {
                return AI_ERR_RUNTIME;
            }
        }

        rec_model_bytes_.assign(
            static_cast<const uint8_t*>(rec_model_data),
            static_cast<const uint8_t*>(rec_model_data) + rec_model_size);
        rec_session_ = create_session_from_memory(rec_model_bytes_.data(), rec_model_bytes_.size(), config_, error);
        loaded_without_model_ = false;
        if (!rec_session_) {
            return AI_ERR_RUNTIME;
        }

        read_ocr_config();
        if (det_session_) {
            cache_node_names(*det_session_, det_input_names_, det_output_names_);
        }
        cache_node_names(*rec_session_, rec_input_names_, rec_output_names_);
        return AI_OK;
    }

    // 执行可选文本检测，裁剪每个区域，识别文本，并填充 C ABI 行结果。
    int32_t recognize(
        const AIImage& image,
        AIOcrLine* output,
        int32_t max_output,
        OcrRecognitionDiagnostics* diagnostics) override {
        if (diagnostics != nullptr) *diagnostics = OcrRecognitionDiagnostics{};
        if (!rec_session_) {
            return loaded_without_model_ ? AI_ERR_BACKEND_NOT_CONFIGURED : AI_ERR_RUNTIME;
        }
        if (charset_.empty()) {
            return AI_ERR_CONFIG;
        }
        if (output == nullptr || max_output <= 0) {
            return AI_ERR_INVALID_ARGUMENT;
        }
        try {
            stage_latency_us_.fill(0);
            const int channels = channels_for_format(image.format);
            if (channels == 0) {
                return AI_ERR_IMAGE_FORMAT;
            }

            // 收集需要识别的文本区域。
            text_boxes_.clear();
            size_t merged_hypothesis_box_count = 0;
            bool compare_split_box_hypothesis = false;

            // A. 文本检测：当 det_session_ 存在且 rec_only_ == false 时执行。
            if (det_session_ && !rec_only_) {
                if (diagnostics != nullptr) diagnostics->detection_performed = true;
                // 检测前处理遵循 PP-OCRv6 导出配置：BGR、ImageNet mean/std、动态尺寸 32 对齐。
                det_input_data_.clear();
                // 使用检测 session 中的模型输入尺寸。
                auto det_info = det_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
                auto det_shape = det_info.GetShape();
                const int model_h = (det_shape.size() >= 3 && det_shape[2] > 0) ? static_cast<int>(det_shape[2]) : 0;
                const int model_w = (det_shape.size() >= 4 && det_shape[3] > 0) ? static_cast<int>(det_shape[3]) : 0;
                if ((det_input_height_ > 0 && model_h > 0 && det_input_height_ != model_h) ||
                    (det_input_width_ > 0 && model_w > 0 && det_input_width_ != model_w)) {
                    return AI_ERR_INVALID_ARGUMENT;
                }
                const auto align32 = [](int value) {
                    return std::max(32, static_cast<int>((static_cast<int64_t>(value) + 16) / 32 * 32));
                };
                const int det_h = det_input_height_ > 0 ? det_input_height_ :
                    (model_h > 0 ? model_h : align32(image.height));
                const int det_w = det_input_width_ > 0 ? det_input_width_ :
                    (model_w > 0 ? model_w : align32(image.width));
                if (det_h <= 0 || det_w <= 0 ||
                    static_cast<int64_t>(det_h) * det_w > std::numeric_limits<int32_t>::max()) {
                    return AI_ERR_INVALID_ARGUMENT;
                }

                ocr_detection_to_chw_float(image, det_w, det_h, det_input_data_);

                const std::array<int64_t, 4> det_input_shape = {1, 3, det_h, det_w};
                Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value det_input_val = Ort::Value::CreateTensor<float>(
                    mem_info, det_input_data_.data(), det_input_data_.size(),
                    det_input_shape.data(), det_input_shape.size());

                std::vector<const char*> det_in_ptrs, det_out_ptrs;
                for (const auto& n : det_input_names_) det_in_ptrs.push_back(n.c_str());
                for (const auto& n : det_output_names_) det_out_ptrs.push_back(n.c_str());

                const auto det_start = std::chrono::steady_clock::now();
                auto det_outputs = det_session_->Run(
                    Ort::RunOptions{nullptr},
                    det_in_ptrs.data(), &det_input_val, det_in_ptrs.size(),
                    det_out_ptrs.data(), det_out_ptrs.size());
                stage_latency_us_[AI_OCR_STAGE_DETECTION] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - det_start).count();

                if (!det_outputs.empty()) {
                    const auto post_start = std::chrono::steady_clock::now();
                    const auto& det_out = det_outputs[0];
                    auto det_out_shape = det_out.GetTensorTypeAndShapeInfo().GetShape();
                    const int out_h = (det_out_shape.size() >= 3) ? static_cast<int>(det_out_shape[2]) : det_h;
                    const int out_w = (det_out_shape.size() >= 4) ? static_cast<int>(det_out_shape[3]) : det_w;
                    const float* prob_map = det_out.GetTensorData<float>();

                    auto raw_boxes = find_text_boxes(
                        prob_map,
                        out_w,
                        out_h,
                        det_binary_threshold_,
                        det_min_area_,
                        det_box_score_threshold_,
                        det_unclip_ratio_,
                        false);
                    std::vector<TextBox> diagonal_boxes;
                    // The second connectivity pass is an abnormal-path tool.
                    // Avoid rescanning a large probability map on the stable
                    // multi-line fast path, where it cannot be selected.
                    if (image.height <= 64 && raw_boxes.size() >= 3) {
                        diagonal_boxes = find_text_boxes(
                            prob_map,
                            out_w,
                            out_h,
                            det_binary_threshold_,
                            det_min_area_,
                            det_box_score_threshold_,
                            det_unclip_ratio_,
                            true);
                    }
                    if (diagnostics != nullptr) {
                        diagnostics->detected_box_count = static_cast<int32_t>(raw_boxes.size());
                        diagnostics->four_connected_box_count =
                            static_cast<int32_t>(raw_boxes.size());
                        diagnostics->eight_connected_box_count =
                            static_cast<int32_t>(diagonal_boxes.size());
                        diagnostics->detection_empty = raw_boxes.empty();
                    }

                    // Eight-connectivity is a no-inference-cost alternative
                    // from the same probability map. Use it only when both
                    // hypotheses are a verified small single line and it
                    // strictly reduces a fragmented four-connected result.
                    // All other panels retain the established geometry.
                    const bool use_diagonal_boxes = image.height <= 64 &&
                        raw_boxes.size() >= 3 && !diagonal_boxes.empty() &&
                        diagonal_boxes.size() < raw_boxes.size() &&
                        group_ocr_reading_order(raw_boxes).size() == 1 &&
                        group_ocr_reading_order(diagonal_boxes).size() == 1;
                    if (use_diagonal_boxes) {
                        raw_boxes = std::move(diagonal_boxes);
                        if (diagnostics != nullptr) {
                            diagnostics->used_eight_connectivity = true;
                        }
                    }

                    // 将检测框映射回原图坐标。
                    for (auto& box : raw_boxes) {
                        const float scale_x = static_cast<float>(image.width) / out_w;
                        const float scale_y = static_cast<float>(image.height) / out_h;
                        float x = box.x * scale_x;
                        float y = box.y * scale_y;
                        float w = box.w * scale_x;
                        float h = box.h * scale_y;
                        // 裁剪边界。
                        int bx = std::max(0, static_cast<int>(x));
                        int by = std::max(0, static_cast<int>(y));
                        int bw = std::min(image.width - bx, static_cast<int>(w + 0.5f));
                        int bh = std::min(image.height - by, static_cast<int>(h + 0.5f));
                        if (bw > 0 && bh > 0) {
                            text_boxes_.push_back(TextBox{bx, by, bw, bh});
                        }
                    }

                    sort_ocr_reading_order(&text_boxes_);
                    bool merged_any = false;
                    size_t merged_fragment_count = 0;
                    std::vector<TextBox> merged_boxes = text_boxes_;
                    // Whole-image OCR panels already have stable multi-line
                    // geometry. Restrict fragment joining to a geometrically
                    // verified small single line, where overlapping recognition
                    // crops are the observed duplication mechanism.
                    if (image.height <= 64 && text_boxes_.size() >= 3) {
                        const auto reading_lines =
                            group_ocr_reading_order(text_boxes_);
                        if (reading_lines.size() == 1) {
                            merged_boxes = merge_fragmented_ocr_boxes(
                                text_boxes_,
                                rec_horizontal_padding_ratio_,
                                &merged_any,
                                &merged_fragment_count);
                        }
                    }
                    if (merged_any) {
                        // The established split boxes remain a recognition
                        // hypothesis.  Put the joint boxes first so callers
                        // with a tight output capacity still receive the
                        // structurally complete result; the split result is
                        // retained in internal diagnostics for selection.
                        if (rec_batch_size_ == 1 && diagnostics != nullptr) {
                            const std::vector<TextBox> split_boxes = text_boxes_;
                            merged_hypothesis_box_count = merged_boxes.size();
                            merged_boxes.insert(
                                merged_boxes.end(),
                                split_boxes.begin(),
                                split_boxes.end());
                            compare_split_box_hypothesis = true;
                        }
                        text_boxes_ = std::move(merged_boxes);
                        if (diagnostics != nullptr) {
                            diagnostics->used_merged_line_recognition = true;
                            diagnostics->merged_fragment_count =
                                static_cast<int32_t>(std::min<size_t>(
                                    merged_fragment_count,
                                    static_cast<size_t>(
                                        std::numeric_limits<int32_t>::max())));
                        }
                    }
                    stage_latency_us_[AI_OCR_STAGE_POSTPROCESS] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - post_start).count();
                }
            }

            // 检测模式下“没有文本框”就是正常的空结果，不能把多行截图伪装成
            // 单行送入识别模型。只有纯识别模式才允许整图识别。
            if (text_boxes_.empty() && (!det_session_ || rec_only_)) {
                text_boxes_.push_back(TextBox{0, 0, image.width, image.height});
                if (diagnostics != nullptr) diagnostics->whole_image_recognition = true;
            }
            if (text_boxes_.empty()) {
                if (diagnostics != nullptr) {
                    diagnostics->detection_us = stage_latency_us_[AI_OCR_STAGE_DETECTION];
                    diagnostics->recognition_us = 0;
                    diagnostics->postprocess_us = stage_latency_us_[AI_OCR_STAGE_POSTPROCESS];
                }
                return 0;
            }

            // B. 对每个区域执行文本识别。
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<const char*> rec_in_ptrs, rec_out_ptrs;
            for (const auto& n : rec_input_names_) rec_in_ptrs.push_back(n.c_str());
            for (const auto& n : rec_output_names_) rec_out_ptrs.push_back(n.c_str());

            const auto rec_start = std::chrono::steady_clock::now();
            int32_t result_count = 0;
            const bool is_bgr = (image.format == AI_IMAGE_BGR24 || image.format == AI_IMAGE_BGRA32);
            const auto rec_model_shape = rec_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            const int rec_model_width = rec_model_shape.size() >= 4 && rec_model_shape[3] > 0
                ? static_cast<int>(rec_model_shape[3]) : 0;

            // 正式路径使用逐框识别。该 PP-OCRv6 模型使用动态 batch 会产生
            // 显著长尾；批处理实现仅在内部配置显式设置
            // ocr.rec_batch_size > 1 时启用。
            if (rec_batch_size_ == 1) {
                int32_t merged_hypothesis_result_count = -1;
                for (size_t recognition_box_index = 0;
                     recognition_box_index < text_boxes_.size();
                     ++recognition_box_index) {
                    if (compare_split_box_hypothesis &&
                        recognition_box_index == merged_hypothesis_box_count) {
                        merged_hypothesis_result_count = result_count;
                    }
                    if (result_count >= max_output) break;
                    const auto& box = text_boxes_[recognition_box_index];
                    const int horizontal_padding = std::max(
                        1,
                        static_cast<int>(std::ceil(
                            box.h * rec_horizontal_padding_ratio_)));
                    const int top_padding = std::max(
                        0,
                        static_cast<int>(std::ceil(
                            box.h * rec_top_padding_ratio_)));
                    const int bottom_padding = std::max(
                        0,
                        static_cast<int>(std::ceil(
                            box.h * rec_bottom_padding_ratio_)));
                    const int crop_x =
                        std::max(0, box.x - horizontal_padding);
                    const int crop_y =
                        std::max(0, box.y - top_padding);
                    const int crop_right = std::min(
                        image.width,
                        box.x + box.w + horizontal_padding);
                    const int crop_bottom = std::min(
                        image.height,
                        box.y + box.h + bottom_padding);
                    const int crop_w = crop_right - crop_x;
                    const int crop_h = crop_bottom - crop_y;
                    const float ratio =
                        static_cast<float>(input_height_) /
                        static_cast<float>(crop_h);
                    int target_w =
                        static_cast<int>(crop_w * ratio + 0.5f);
                    target_w = std::max(target_w, 1);
                    const int tensor_w = rec_model_width > 0
                        ? rec_model_width
                        : std::max(
                              input_width_,
                              static_cast<int>(
                                  (static_cast<int64_t>(target_w) + 31) /
                                  32 * 32));
                    target_w = std::min(target_w, tensor_w);
                    if (tensor_w <= 0 ||
                        static_cast<int64_t>(input_height_) * tensor_w >
                            std::numeric_limits<int32_t>::max()) {
                        return AI_ERR_INVALID_ARGUMENT;
                    }
                    const size_t plane_size =
                        static_cast<size_t>(input_height_) * tensor_w;
                    rec_input_data_.assign(3 * plane_size, 0.0f);
                    for (int dy = 0; dy < input_height_; ++dy) {
                        const float source_y = std::max(
                            0.0f, (dy + 0.5f) / ratio - 0.5f);
                        const int sy0 = std::min(
                            static_cast<int>(source_y), crop_h - 1);
                        const int sy1 = std::min(sy0 + 1, crop_h - 1);
                        const float fy = source_y - sy0;
                        const uint8_t* row0 =
                            image_row_ptr(image, crop_y + sy0);
                        const uint8_t* row1 =
                            image_row_ptr(image, crop_y + sy1);
                        for (int dx = 0; dx < target_w; ++dx) {
                            const float source_x = std::max(
                                0.0f,
                                (dx + 0.5f) / ratio - 0.5f);
                            const int sx0 = std::min(
                                static_cast<int>(source_x), crop_w - 1);
                            const int sx1 =
                                std::min(sx0 + 1, crop_w - 1);
                            const float fx = source_x - sx0;
                            float values[3]{};
                            for (int c = 0; c < 3; ++c) {
                                const int source_channel =
                                    channels == 1 ? 0 : c;
                                const float top =
                                    row0[
                                        (crop_x + sx0) * channels +
                                        source_channel] *
                                        (1.0f - fx) +
                                    row0[
                                        (crop_x + sx1) * channels +
                                        source_channel] *
                                        fx;
                                const float bottom =
                                    row1[
                                        (crop_x + sx0) * channels +
                                        source_channel] *
                                        (1.0f - fx) +
                                    row1[
                                        (crop_x + sx1) * channels +
                                        source_channel] *
                                        fx;
                                values[c] =
                                    (top * (1.0f - fy) + bottom * fy) /
                                    255.0f;
                            }
                            const float b_val = channels == 1
                                ? values[0]
                                : (is_bgr ? values[0] : values[2]);
                            const float g_val =
                                values[channels == 1 ? 0 : 1];
                            const float r_val = channels == 1
                                ? values[0]
                                : (is_bgr ? values[2] : values[0]);
                            const size_t pixel_idx =
                                static_cast<size_t>(dy) * tensor_w + dx;
                            if (rec_channel_order_bgr_) {
                                rec_input_data_[
                                    pixel_idx] =
                                    (b_val - 0.5f) / 0.5f;
                                rec_input_data_[
                                    plane_size + pixel_idx] =
                                    (g_val - 0.5f) / 0.5f;
                                rec_input_data_[
                                    2 * plane_size + pixel_idx] =
                                    (r_val - 0.5f) / 0.5f;
                            } else {
                                rec_input_data_[
                                    pixel_idx] =
                                    (r_val - 0.5f) / 0.5f;
                                rec_input_data_[
                                    plane_size + pixel_idx] =
                                    (g_val - 0.5f) / 0.5f;
                                rec_input_data_[
                                    2 * plane_size + pixel_idx] =
                                    (b_val - 0.5f) / 0.5f;
                            }
                        }
                    }

                    const std::array<int64_t, 4> rec_shape = {
                        1,
                        3,
                        static_cast<int64_t>(input_height_),
                        static_cast<int64_t>(tensor_w)};
                    Ort::Value rec_input_val =
                        Ort::Value::CreateTensor<float>(
                            mem_info,
                            rec_input_data_.data(),
                            rec_input_data_.size(),
                            rec_shape.data(),
                            rec_shape.size());
                    auto rec_outputs = rec_session_->Run(
                        Ort::RunOptions{nullptr},
                        rec_in_ptrs.data(),
                        &rec_input_val,
                        rec_in_ptrs.size(),
                        rec_out_ptrs.data(),
                        rec_out_ptrs.size());
                    if (rec_outputs.empty()) continue;
                    const auto& rec_out = rec_outputs[0];
                    const auto rec_out_shape =
                        rec_out.GetTensorTypeAndShapeInfo().GetShape();
                    if (rec_out_shape.size() < 2) continue;
                    int seq_len = 0;
                    int num_classes = 0;
                    bool classes_last = true;
                    if (rec_out_shape.size() >= 3) {
                        const int dim1 =
                            static_cast<int>(rec_out_shape[1]);
                        const int dim2 =
                            static_cast<int>(rec_out_shape[2]);
                        const int expected_classes =
                            static_cast<int>(charset_.size());
                        const bool dim1_matches_classes =
                            expected_classes > 1 &&
                            std::abs(dim1 - expected_classes) <=
                                std::abs(dim2 - expected_classes);
                        classes_last = !dim1_matches_classes;
                        seq_len = classes_last ? dim1 : dim2;
                        num_classes = classes_last ? dim2 : dim1;
                    } else {
                        seq_len =
                            static_cast<int>(rec_out_shape[0]);
                        num_classes =
                            static_cast<int>(rec_out_shape[1]);
                    }
                    if (seq_len <= 0 || num_classes <= 0) continue;
                    const float* logits =
                        rec_out.GetTensorData<float>();
                    if (!classes_last) {
                        transposed_logits_.resize(
                            static_cast<size_t>(seq_len) *
                            num_classes);
                        for (int t = 0; t < seq_len; ++t) {
                            for (int c = 0; c < num_classes; ++c) {
                                transposed_logits_[
                                    static_cast<size_t>(t) *
                                        num_classes +
                                    c] =
                                    logits[
                                        static_cast<size_t>(c) *
                                            seq_len +
                                        t];
                            }
                        }
                        logits = transposed_logits_.data();
                    }
                    CtcResult ctc = decoder_ == "beam"
                        ? ctc_beam_decode(
                              logits,
                              seq_len,
                              num_classes,
                              charset_,
                              beam_width_)
                        : ctc_greedy_decode(
                              logits,
                              seq_len,
                              num_classes,
                              charset_);
                    normalize_ocr_punctuation(&ctc.text);
                    if (ctc.text.empty()) continue;
                    AIOcrLine& out_line = output[result_count];
                    out_line.box.x = box.x;
                    out_line.box.y = box.y;
                    out_line.box.w = box.w;
                    out_line.box.h = box.h;
                    out_line.confidence = ctc.confidence;
                    size_t copy_len = std::min(
                        ctc.text.size(),
                        static_cast<size_t>(
                            AIENGINE_MAX_TEXT - 1));
                    while (
                        copy_len > 0 &&
                        copy_len < ctc.text.size() &&
                        (static_cast<unsigned char>(
                             ctc.text[copy_len]) &
                         0xC0u) == 0x80u) {
                        --copy_len;
                    }
                    std::memcpy(
                        out_line.text,
                        ctc.text.c_str(),
                        copy_len);
                    out_line.text[copy_len] = '\0';
                    ++result_count;
                }
                stage_latency_us_[AI_OCR_STAGE_RECOGNITION] =
                    std::chrono::duration_cast<
                        std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - rec_start)
                        .count();
                if (diagnostics != nullptr) {
                    if (compare_split_box_hypothesis) {
                        if (merged_hypothesis_result_count < 0) {
                            merged_hypothesis_result_count = result_count;
                        }
                        diagnostics->split_box_lines.assign(
                            output + merged_hypothesis_result_count,
                            output + result_count);
                        result_count = merged_hypothesis_result_count;
                    }
                    diagnostics->detection_us = stage_latency_us_[AI_OCR_STAGE_DETECTION];
                    diagnostics->recognition_us = stage_latency_us_[AI_OCR_STAGE_RECOGNITION];
                    diagnostics->postprocess_us = stage_latency_us_[AI_OCR_STAGE_POSTPROCESS];
                }
                return result_count;
            }

            // 先计算所有裁剪和张量宽度，再按相同宽度分组。这样批次内不改变
            // 任意文字框的缩放或 padding，CPU/DirectML 的公开识别语义保持不变。
            rec_regions_.clear();
            rec_regions_.reserve(text_boxes_.size());
            for (size_t box_index = 0; box_index < text_boxes_.size(); ++box_index) {
                const TextBox& box = text_boxes_[box_index];
                const int horizontal_padding = std::max(
                    1, static_cast<int>(std::ceil(
                           box.h * rec_horizontal_padding_ratio_)));
                const int top_padding = std::max(
                    0, static_cast<int>(std::ceil(
                           box.h * rec_top_padding_ratio_)));
                const int bottom_padding = std::max(
                    0, static_cast<int>(std::ceil(
                           box.h * rec_bottom_padding_ratio_)));
                OcrRecRegion region;
                region.box = box;
                region.crop_x = std::max(0, box.x - horizontal_padding);
                region.crop_y = std::max(0, box.y - top_padding);
                const int crop_right = std::min(
                    image.width, box.x + box.w + horizontal_padding);
                const int crop_bottom = std::min(
                    image.height, box.y + box.h + bottom_padding);
                region.crop_w = crop_right - region.crop_x;
                region.crop_h = crop_bottom - region.crop_y;
                if (region.crop_w <= 0 || region.crop_h <= 0) continue;
                const float ratio =
                    static_cast<float>(input_height_) / region.crop_h;
                region.target_w = std::max(
                    1, static_cast<int>(region.crop_w * ratio + 0.5f));
                region.tensor_w = rec_model_width > 0
                    ? rec_model_width
                    : std::max(
                          input_width_,
                          static_cast<int>(
                              (static_cast<int64_t>(region.target_w) + 31) /
                              32 * 32));
                region.target_w =
                    std::min(region.target_w, region.tensor_w);
                if (region.tensor_w <= 0 ||
                    static_cast<int64_t>(input_height_) * region.tensor_w >
                        std::numeric_limits<int32_t>::max()) {
                    return AI_ERR_INVALID_ARGUMENT;
                }
                region.output_index = box_index;
                rec_regions_.push_back(region);
            }

            rec_region_order_.resize(rec_regions_.size());
            std::iota(
                rec_region_order_.begin(), rec_region_order_.end(), size_t{0});
            std::stable_sort(
                rec_region_order_.begin(),
                rec_region_order_.end(),
                [this](size_t lhs, size_t rhs) {
                    return rec_regions_[lhs].tensor_w <
                           rec_regions_[rhs].tensor_w;
                });
            rec_decoded_regions_.assign(
                rec_regions_.size(), OcrDecodedRegion{});

            const bool dynamic_batch =
                rec_model_shape.empty() || rec_model_shape[0] <= 0;
            const size_t batch_limit = dynamic_batch
                ? static_cast<size_t>(rec_batch_size_)
                : size_t{1};

            std::string rec_failure;
            const auto execute_rec_batch =
                [&](size_t order_begin, size_t batch_count) -> bool {
                if (batch_count == 0) return true;
                const int tensor_w =
                    rec_regions_[rec_region_order_[order_begin]].tensor_w;
                const size_t plane_size =
                    static_cast<size_t>(input_height_) * tensor_w;
                const size_t sample_size = 3 * plane_size;
                rec_input_data_.assign(
                    batch_count * sample_size, 0.0f);

                for (size_t batch_index = 0;
                     batch_index < batch_count;
                     ++batch_index) {
                    const OcrRecRegion& region =
                        rec_regions_[rec_region_order_[
                            order_begin + batch_index]];
                    const float ratio =
                        static_cast<float>(input_height_) / region.crop_h;
                    const size_t sample_offset =
                        batch_index * sample_size;
                    for (int dy = 0; dy < input_height_; ++dy) {
                        const float source_y = std::max(
                            0.0f, (dy + 0.5f) / ratio - 0.5f);
                        const int sy0 = std::min(
                            static_cast<int>(source_y),
                            region.crop_h - 1);
                        const int sy1 =
                            std::min(sy0 + 1, region.crop_h - 1);
                        const float fy = source_y - sy0;
                        const uint8_t* row0 = image_row_ptr(
                            image, region.crop_y + sy0);
                        const uint8_t* row1 = image_row_ptr(
                            image, region.crop_y + sy1);
                        for (int dx = 0; dx < region.target_w; ++dx) {
                            const float source_x = std::max(
                                0.0f, (dx + 0.5f) / ratio - 0.5f);
                            const int sx0 = std::min(
                                static_cast<int>(source_x),
                                region.crop_w - 1);
                            const int sx1 =
                                std::min(sx0 + 1, region.crop_w - 1);
                            const float fx = source_x - sx0;
                            float values[3]{};
                            for (int c = 0; c < 3; ++c) {
                                const int source_channel =
                                    channels == 1 ? 0 : c;
                                const int left =
                                    (region.crop_x + sx0) * channels +
                                    source_channel;
                                const int right =
                                    (region.crop_x + sx1) * channels +
                                    source_channel;
                                const float top =
                                    row0[left] * (1.0f - fx) +
                                    row0[right] * fx;
                                const float bottom =
                                    row1[left] * (1.0f - fx) +
                                    row1[right] * fx;
                                values[c] =
                                    (top * (1.0f - fy) + bottom * fy) /
                                    255.0f;
                            }
                            const float b_val = channels == 1
                                ? values[0]
                                : (is_bgr ? values[0] : values[2]);
                            const float g_val =
                                values[channels == 1 ? 0 : 1];
                            const float r_val = channels == 1
                                ? values[0]
                                : (is_bgr ? values[2] : values[0]);
                            const size_t pixel_idx =
                                static_cast<size_t>(dy) * tensor_w + dx;
                            const float first = rec_channel_order_bgr_
                                ? b_val
                                : r_val;
                            const float third = rec_channel_order_bgr_
                                ? r_val
                                : b_val;
                            rec_input_data_[
                                sample_offset + pixel_idx] =
                                (first - 0.5f) / 0.5f;
                            rec_input_data_[
                                sample_offset + plane_size + pixel_idx] =
                                (g_val - 0.5f) / 0.5f;
                            rec_input_data_[
                                sample_offset + 2 * plane_size + pixel_idx] =
                                (third - 0.5f) / 0.5f;
                        }
                    }
                }

                try {
                    const std::array<int64_t, 4> rec_shape = {
                        static_cast<int64_t>(batch_count),
                        3,
                        static_cast<int64_t>(input_height_),
                        static_cast<int64_t>(tensor_w)};
                    Ort::Value rec_input_val =
                        Ort::Value::CreateTensor<float>(
                            mem_info,
                            rec_input_data_.data(),
                            rec_input_data_.size(),
                            rec_shape.data(),
                            rec_shape.size());
                    auto rec_outputs = rec_session_->Run(
                        Ort::RunOptions{nullptr},
                        rec_in_ptrs.data(),
                        &rec_input_val,
                        rec_in_ptrs.size(),
                        rec_out_ptrs.data(),
                        rec_out_ptrs.size());
                    if (rec_outputs.empty()) {
                        rec_failure =
                            "OCR recognition session returned no outputs";
                        return false;
                    }

                    const auto& rec_out = rec_outputs[0];
                    const auto rec_out_info =
                        rec_out.GetTensorTypeAndShapeInfo();
                    const auto rec_out_shape = rec_out_info.GetShape();
                    if (rec_out_shape.size() < 2) {
                        rec_failure =
                            "OCR recognition output rank is less than 2";
                        return false;
                    }
                    int seq_len = 0;
                    int num_classes = 0;
                    bool classes_last = true;
                    if (rec_out_shape.size() >= 3) {
                        const int dim1 =
                            static_cast<int>(rec_out_shape[1]);
                        const int dim2 =
                            static_cast<int>(rec_out_shape[2]);
                        const int expected_classes =
                            static_cast<int>(charset_.size());
                        const bool dim1_matches_classes =
                            expected_classes > 1 &&
                            std::abs(dim1 - expected_classes) <=
                                std::abs(dim2 - expected_classes);
                        classes_last = !dim1_matches_classes;
                        seq_len = classes_last ? dim1 : dim2;
                        num_classes = classes_last ? dim2 : dim1;
                    } else if (batch_count == 1) {
                        seq_len =
                            static_cast<int>(rec_out_shape[0]);
                        num_classes =
                            static_cast<int>(rec_out_shape[1]);
                    } else {
                        rec_failure =
                            "OCR recognition batched output shape is invalid";
                        return false;
                    }
                    if (seq_len <= 0 || num_classes <= 0) {
                        rec_failure =
                            "OCR recognition output has an invalid sequence or class dimension";
                        return false;
                    }
                    const size_t output_stride =
                        static_cast<size_t>(seq_len) * num_classes;
                    if (rec_out_info.GetElementCount() <
                        output_stride * batch_count) {
                        rec_failure =
                            "OCR recognition output tensor is smaller than its declared batch shape";
                        return false;
                    }

                    const float* all_logits =
                        rec_out.GetTensorData<float>();
                    if (!classes_last) {
                        transposed_logits_.resize(
                            output_stride * batch_count);
                        for (size_t b = 0; b < batch_count; ++b) {
                            for (int t = 0; t < seq_len; ++t) {
                                for (int c = 0; c < num_classes; ++c) {
                                    transposed_logits_[
                                        b * output_stride +
                                        static_cast<size_t>(t) *
                                            num_classes +
                                        c] =
                                        all_logits[
                                            b * output_stride +
                                            static_cast<size_t>(c) *
                                                seq_len +
                                            t];
                                }
                            }
                        }
                        all_logits = transposed_logits_.data();
                    }

                    for (size_t batch_index = 0;
                         batch_index < batch_count;
                         ++batch_index) {
                        OcrDecodedRegion& decoded =
                            rec_decoded_regions_[rec_region_order_[
                                order_begin + batch_index]];
                        const float* logits =
                            all_logits + batch_index * output_stride;
                        decoded.ctc = decoder_ == "beam"
                            ? ctc_beam_decode(
                                  logits,
                                  seq_len,
                                  num_classes,
                                  charset_,
                                  beam_width_)
                            : ctc_greedy_decode(
                                  logits,
                                  seq_len,
                                  num_classes,
                                  charset_);
                        normalize_ocr_punctuation(&decoded.ctc.text);
                        decoded.valid = true;
                    }
                    return true;
                } catch (const Ort::Exception& ex) {
                    rec_failure =
                        std::string("ONNX Runtime OCR recognition failed: ") +
                        ex.what();
                    return false;
                }
            };

            for (size_t order_begin = 0;
                 order_begin < rec_region_order_.size();) {
                const int tensor_w =
                    rec_regions_[rec_region_order_[order_begin]].tensor_w;
                size_t batch_count = 1;
                while (batch_count < batch_limit &&
                       order_begin + batch_count <
                           rec_region_order_.size() &&
                       rec_regions_[rec_region_order_[
                           order_begin + batch_count]].tensor_w ==
                           tensor_w) {
                    ++batch_count;
                }
                if (!execute_rec_batch(order_begin, batch_count)) {
                    if (batch_count == 1) {
                        ai::set_last_error(
                            rec_failure.empty()
                                ? "OCR recognition session failed without details"
                                : rec_failure);
                        return AI_ERR_RUNTIME;
                    }
                    for (size_t offset = 0;
                         offset < batch_count;
                         ++offset) {
                        if (!execute_rec_batch(
                                order_begin + offset, 1)) {
                            ai::set_last_error(
                                rec_failure.empty()
                                    ? "OCR recognition session failed without details"
                                    : rec_failure);
                            return AI_ERR_RUNTIME;
                        }
                    }
                }
                order_begin += batch_count;
            }

            // 按检测后的阅读顺序写回，分组仅影响执行顺序。
            std::vector<size_t> output_order(rec_regions_.size());
            std::iota(
                output_order.begin(), output_order.end(), size_t{0});
            std::stable_sort(
                output_order.begin(),
                output_order.end(),
                [this](size_t lhs, size_t rhs) {
                    return rec_regions_[lhs].output_index <
                           rec_regions_[rhs].output_index;
                });
            for (const size_t region_index : output_order) {
                if (result_count >= max_output) break;
                const OcrDecodedRegion& decoded =
                    rec_decoded_regions_[region_index];
                if (!decoded.valid || decoded.ctc.text.empty()) continue;
                const TextBox& box = rec_regions_[region_index].box;
                AIOcrLine& out_line = output[result_count];
                out_line.box.x = box.x;
                out_line.box.y = box.y;
                out_line.box.w = box.w;
                out_line.box.h = box.h;
                out_line.confidence = decoded.ctc.confidence;
                size_t copy_len = std::min(
                    decoded.ctc.text.size(),
                    static_cast<size_t>(AIENGINE_MAX_TEXT - 1));
                while (
                    copy_len > 0 &&
                    copy_len < decoded.ctc.text.size() &&
                    (static_cast<unsigned char>(
                         decoded.ctc.text[copy_len]) &
                     0xC0u) == 0x80u) {
                    --copy_len;
                }
                std::memcpy(
                    out_line.text,
                    decoded.ctc.text.c_str(),
                    copy_len);
                out_line.text[copy_len] = '\0';
                ++result_count;
            }

            stage_latency_us_[AI_OCR_STAGE_RECOGNITION] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - rec_start).count();
            if (diagnostics != nullptr) {
                diagnostics->detection_us = stage_latency_us_[AI_OCR_STAGE_DETECTION];
                diagnostics->recognition_us = stage_latency_us_[AI_OCR_STAGE_RECOGNITION];
                diagnostics->postprocess_us = stage_latency_us_[AI_OCR_STAGE_POSTPROCESS];
            }
            return result_count;

        } catch (const Ort::Exception& ex) {
            ai::set_last_error(
                std::string("ONNX Runtime OCR inference failed: ") +
                ex.what());
            return AI_ERR_RUNTIME;
        } catch (const std::exception& ex) {
            ai::set_last_error(
                std::string("OCR inference failed: ") + ex.what());
            return AI_ERR_RUNTIME;
        }
    }

    int64_t ocr_stage_latency_us(int32_t stage) const override {
        if (stage < AI_OCR_STAGE_DETECTION || stage > AI_OCR_STAGE_POSTPROCESS) return -1;
        return stage_latency_us_[stage];
    }

    // 释放所有 ORT session、缓存节点名、字符集和模型字节。
    int32_t release_models() override {
        det_session_.reset();
        rec_session_.reset();
        det_model_bytes_.clear();
        rec_model_bytes_.clear();
        charset_.clear();
        det_input_names_.clear();
        det_output_names_.clear();
        rec_input_names_.clear();
        rec_output_names_.clear();
        loaded_without_model_ = false;
        return AI_OK;
    }

private:
    // 读取识别尺寸、通道顺序、纯识别模式和字符字典。
    void read_ocr_config() {
        input_height_ = config_.get_int("ocr.input_height", 48);
        input_width_ = config_.get_int("ocr.input_width", 320);
        rec_only_ = config_.get_bool("ocr.rec_only", true);
        const std::string channel_order = to_lower_ascii(config_.get_string("ocr.channel_order", "bgr"));
        rec_channel_order_bgr_ = (channel_order != "rgb");
        det_input_height_ = config_.get_int("ocr.det_input_height", 0);
        det_input_width_ = config_.get_int("ocr.det_input_width", 0);
        det_binary_threshold_ = config_.get_float("ocr.det_binary_threshold", 0.3f);
        det_box_score_threshold_ = config_.get_float("ocr.det_box_score_threshold", 0.0f);
        det_unclip_ratio_ = config_.get_float("ocr.det_unclip_ratio", 1.0f);
        det_min_area_ = std::max(1, config_.get_int("ocr.det_min_area", 50));
        decoder_ = to_lower_ascii(config_.get_string("ocr.decoder", "greedy"));
        beam_width_ = std::max(2, std::min(16, config_.get_int("ocr.beam_width", 5)));
        rec_batch_size_ = std::max(1, std::min(8, config_.get_int("ocr.rec_batch_size", 1)));
        rec_horizontal_padding_ratio_ = std::max(0.0f, std::min(1.0f, config_.get_float("ocr.rec_horizontal_padding_ratio", 0.25f)));
        rec_top_padding_ratio_ = std::max(0.0f, std::min(0.5f, config_.get_float("ocr.rec_top_padding_ratio", 0.05f)));
        rec_bottom_padding_ratio_ = std::max(0.0f, std::min(0.5f, config_.get_float("ocr.rec_bottom_padding_ratio", 0.0f)));

        const std::string charset_path = config_.get_string("ocr.charset_path", "");
        const std::string charset_inline = config_.get_string("ocr.charset_inline", "");
        if (!charset_inline.empty()) {
            load_lines_from_inline_text(charset_inline, charset_, true);
        } else if (!charset_path.empty()) {
            load_charset_file(charset_path, charset_);
        }
        // PaddleOCR 的 CTCLabelDecode 在自定义字典后默认追加空格，再在索引 0 插入 blank。
        if (!charset_.empty() && charset_.back() != " ") {
            charset_.push_back(" ");
        }
    }

    Config config_;
    std::unique_ptr<Ort::Session> det_session_;
    std::unique_ptr<Ort::Session> rec_session_;
    std::vector<uint8_t> det_model_bytes_;
    std::vector<uint8_t> rec_model_bytes_;
    std::vector<TextBox> text_boxes_;
    std::vector<float> det_input_data_;
    std::vector<float> rec_input_data_;
    std::vector<float> transposed_logits_;
    std::vector<OcrRecRegion> rec_regions_;
    std::vector<size_t> rec_region_order_;
    std::vector<OcrDecodedRegion> rec_decoded_regions_;
    bool loaded_without_model_ = false;

    // 字符集：索引 0 = CTC blank，索引 1+ = 实际字符。
    std::vector<std::string> charset_;
    // 配置参数。
    int input_height_ = 48;
    int input_width_ = 320;
    int det_input_height_ = 0;
    int det_input_width_ = 0;
    int det_min_area_ = 50;
    float det_binary_threshold_ = 0.3f;
    float det_box_score_threshold_ = 0.0f;
    float det_unclip_ratio_ = 1.0f;
    std::string decoder_ = "greedy";
    int beam_width_ = 5;
    int rec_batch_size_ = 1;
    float rec_horizontal_padding_ratio_ = 0.25f;
    float rec_top_padding_ratio_ = 0.05f;
    float rec_bottom_padding_ratio_ = 0.0f;
    std::array<int64_t, 4> stage_latency_us_{};
    bool rec_only_ = true;
    bool rec_channel_order_bgr_ = true;
    // 节点名缓存。
    std::vector<std::string> det_input_names_;
    std::vector<std::string> det_output_names_;
    std::vector<std::string> rec_input_names_;
    std::vector<std::string> rec_output_names_;
};

} // namespace

// 通用后端选择器用于创建 ONNX Runtime YOLO 的工厂函数。
std::unique_ptr<YoloBackend> create_onnxruntime_yolo_backend(const Config& config, std::string* error) {
    auto backend = std::make_unique<OnnxRuntimeYoloBackend>(config);
    const int32_t status = backend->load_model(nullptr, config, error);
    if (status < 0 && status != AI_ERR_BACKEND_NOT_CONFIGURED) {
        return nullptr;
    }
    return backend;
}

// 通用后端选择器用于创建 ONNX Runtime OCR 的工厂函数。
std::unique_ptr<OcrBackend> create_onnxruntime_ocr_backend(const Config& config, std::string* error) {
    auto backend = std::make_unique<OnnxRuntimeOcrBackend>(config);
    const int32_t status = backend->load_models(nullptr, nullptr, config, error);
    if (status < 0 && status != AI_ERR_BACKEND_NOT_CONFIGURED) {
        return nullptr;
    }
    return backend;
}

} // namespace ai
