#include "ai_engine.h"
#include "compact_result_format.h"
#include "config.h"
#include "coordinate_offset.h"
#include "embedded_assets.h"
#include "engine.h"
#include "error.h"
#include "runtime_bundle_loader.h"
#include "runtime_status.h"
#include "yolo_pool.h"
#include "worker_protocol.h"
#include "pipe_io.h"
#include "bmp_view.h"
#include "ocr_color_filter.h"
#include <onnxruntime_cxx_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_2.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

static_assert(sizeof(OCRTextResult) == 28, "OCRTextResult wire layout must stay 28 bytes");
constexpr float kYoloInternalNmsThreshold = 0.45f;
using ai::coordinate::offset_ocr_coord_result;
using ai::coordinate::offset_ocr_line;
using ai::coordinate::offset_ocr_lines;
using ai::coordinate::offset_ocr_text_result;
using ai::coordinate::offset_yolo_boxes;

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& data) : data_(data) {}

    bool read_i32(int32_t* out) {
        if (offset_ + sizeof(int32_t) > data_.size() || out == nullptr) return false;
        std::memcpy(out, data_.data() + offset_, sizeof(int32_t));
        offset_ += sizeof(int32_t);
        return true;
    }

    bool read_f32(float* out) {
        if (offset_ + sizeof(float) > data_.size() || out == nullptr) return false;
        std::memcpy(out, data_.data() + offset_, sizeof(float));
        offset_ += sizeof(float);
        return true;
    }

    bool read_u64(uint64_t* out) {
        if (offset_ + sizeof(uint64_t) > data_.size() || out == nullptr) return false;
        std::memcpy(out, data_.data() + offset_, sizeof(uint64_t));
        offset_ += sizeof(uint64_t);
        return true;
    }

    bool read_view(const uint8_t** out, size_t* length) {
        int32_t size = 0;
        if (!out || !length || !read_i32(&size) || size < 0 ||
            static_cast<size_t>(size) > data_.size() - offset_) return false;
        *out = data_.data() + offset_; *length = static_cast<size_t>(size);
        offset_ += static_cast<size_t>(size); return true;
    }

    bool read_bytes(std::vector<uint8_t>* out) {
        int32_t size = 0;
        if (!read_i32(&size) || size < 0 || offset_ + static_cast<size_t>(size) > data_.size() || out == nullptr) return false;
        out->assign(data_.begin() + static_cast<std::ptrdiff_t>(offset_), data_.begin() + static_cast<std::ptrdiff_t>(offset_ + size));
        offset_ += static_cast<size_t>(size);
        return true;
    }

    bool read_string(std::string* out) {
        std::vector<uint8_t> bytes;
        if (!read_bytes(&bytes) || out == nullptr) return false;
        out->assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return true;
    }

    size_t remaining() const { return data_.size() - offset_; }

private:
    const std::vector<uint8_t>& data_;
    size_t offset_ = 0;
};

struct OwnedBmp {
    std::vector<uint8_t> pixels;
    AIImage image{};
};

using YoloSlot = ai::YoloSlot;
using WorkerYoloPool = ai::YoloPool;

enum class WorkerYoloState { Empty, Loading, Loaded, Closing };

struct WorkerYoloContext {
    std::mutex mutex;
    WorkerYoloState state = WorkerYoloState::Empty;
    std::shared_ptr<WorkerYoloPool> pool;
};

using WorkerYoloParams = ai::YoloParameters;

struct OcrSlot {
    std::unique_ptr<ai::Engine> engine;
    bool busy = false;
};

struct OcrPoolContext {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<OcrSlot> slots;
    ai::RuntimeStatus runtime;
};

class OcrLease {
public:
    OcrLease() = default;
    OcrLease(const OcrLease&) = delete;
    OcrLease& operator=(const OcrLease&) = delete;

    ~OcrLease() { release(); }

    int32_t acquire(std::shared_ptr<OcrPoolContext> pool) {
        if (!pool) return AI_ERR_BACKEND_NOT_CONFIGURED;
        std::unique_lock<std::mutex> lock(pool->mutex);
        pool->cv.wait(lock, [&pool] {
            return std::any_of(pool->slots.begin(), pool->slots.end(), [](const OcrSlot& slot) {
                return !slot.busy;
            });
        });
        for (size_t i = 0; i < pool->slots.size(); ++i) {
            if (!pool->slots[i].busy) {
                pool->slots[i].busy = true;
                engine_ = pool->slots[i].engine.get();
                index_ = i;
                pool_ = std::move(pool);
                return AI_OK;
            }
        }
        return AI_ERR_RUNTIME;
    }

    ai::Engine* engine() const { return engine_; }

private:
    void release() {
        if (!pool_) return;
        {
            std::lock_guard<std::mutex> lock(pool_->mutex);
            if (index_ < pool_->slots.size()) pool_->slots[index_].busy = false;
        }
        pool_->cv.notify_one();
        pool_.reset();
        engine_ = nullptr;
    }

    std::shared_ptr<OcrPoolContext> pool_;
    size_t index_ = 0;
    ai::Engine* engine_ = nullptr;
};

std::mutex g_yolo_registry_mutex;
std::unordered_map<int32_t, std::shared_ptr<WorkerYoloContext>> g_yolo_contexts;
int32_t g_next_yolo_handle = 1;
const uint64_t g_worker_instance_id =
    (static_cast<uint64_t>(GetCurrentProcessId()) << 32) ^
    static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
std::mutex g_ocr_registry_mutex;
std::shared_ptr<OcrPoolContext> g_ocr_pool;
std::atomic<int64_t> g_last_latency_us[4]{};
std::atomic<int64_t> g_last_ocr_stage_us[4]{};
std::mutex g_client_mutex;
std::condition_variable g_client_cv;
std::atomic<int32_t> g_active_requests{0};
std::atomic<int32_t> g_live_connections{0};
bool g_shutting_down = false;
constexpr DWORD kClientWatchdogPollMs = 250;
constexpr DWORD kOrphanGraceMs = 3000;
constexpr DWORD kShutdownDrainMs = 5000;
std::unordered_map<DWORD, HANDLE> g_client_processes;
bool g_client_tracking_enabled = false;
std::chrono::steady_clock::time_point g_no_client_since{};
std::atomic<bool> g_runtime_ready{false};
std::string g_runtime_initialization_error;
constexpr int64_t kDirectmlOcrRequestsPerWorker = 900;
constexpr uint64_t kDirectmlOcrPrivateGrowthLimit =
    384ull * 1024ull * 1024ull;
std::atomic<int64_t> g_directml_ocr_request_count{0};
std::atomic<uint64_t> g_directml_ocr_private_baseline{0};
std::atomic<bool> g_ocr_rotation_requested{false};

void register_client_process_id(DWORD process_id) {
    if (process_id == 0) return;
    HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
    if (process == nullptr) return;
    std::lock_guard<std::mutex> lock(g_client_mutex);
    g_client_tracking_enabled = true;
    const auto existing = g_client_processes.find(process_id);
    if (existing == g_client_processes.end()) {
        g_client_processes.emplace(process_id, process);
    } else {
        CloseHandle(process);
    }
    g_no_client_since = std::chrono::steady_clock::time_point{};
}

void register_client_process(HANDLE pipe) {
    ULONG process_id = 0;
    if (pipe != INVALID_HANDLE_VALUE &&
        GetNamedPipeClientProcessId(pipe, &process_id)) {
        register_client_process_id(static_cast<DWORD>(process_id));
    }
}

class ClientGuard {
public:
    ClientGuard() {
        std::lock_guard<std::mutex> lock(g_client_mutex);
        if (!g_shutting_down) {
            ++g_active_requests;
            active_ = true;
        }
    }
    ~ClientGuard() {
        if (!active_) return;
        {
            std::lock_guard<std::mutex> lock(g_client_mutex);
            --g_active_requests;
        }
        g_client_cv.notify_all();
    }
    bool active() const { return active_; }
private:
    bool active_ = false;
};

int32_t read_le_i32(const uint8_t* p) {
    return static_cast<int32_t>(
        static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24));
}

uint16_t read_le_u16(const uint8_t* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_le_u32(const uint8_t* p) {
    return static_cast<uint32_t>(
        static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24));
}

bool parse_bmp24(const uint8_t* data, int32_t size, OwnedBmp* out) {
    if (data == nullptr || size < 54 || out == nullptr) return false;
    if (data[0] != 'B' || data[1] != 'M') return false;
    const uint32_t pixel_offset = read_le_u32(data + 10);
    const uint32_t dib_size = read_le_u32(data + 14);
    if (dib_size < 40 || pixel_offset >= static_cast<uint32_t>(size)) return false;
    const int32_t width = read_le_i32(data + 18);
    const int32_t signed_height = read_le_i32(data + 22);
    const uint16_t planes = read_le_u16(data + 26);
    const uint16_t bit_count = read_le_u16(data + 28);
    const uint32_t compression = read_le_u32(data + 30);
    if (width <= 0 || signed_height == 0 || planes != 1 || bit_count != 24 || compression != 0) return false;

    const int32_t height = signed_height < 0 ? -signed_height : signed_height;
    const bool top_down = signed_height < 0;
    const int32_t src_stride = ((width * 3 + 3) / 4) * 4;
    const uint64_t need = static_cast<uint64_t>(pixel_offset) + static_cast<uint64_t>(src_stride) * static_cast<uint64_t>(height);
    if (need > static_cast<uint64_t>(size)) return false;

    out->pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u, 0);
    for (int32_t y = 0; y < height; ++y) {
        const int32_t src_y = top_down ? y : (height - 1 - y);
        const uint8_t* src = data + pixel_offset + static_cast<size_t>(src_y) * static_cast<size_t>(src_stride);
        uint8_t* dst = out->pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 3u;
        std::memcpy(dst, src, static_cast<size_t>(width) * 3u);
    }
    out->image = AIImage{out->pixels.data(), width, height, width * 3, AI_IMAGE_BGR24};
    return true;
}

void append_i32(std::vector<uint8_t>* out, int32_t value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void append_i64(std::vector<uint8_t>* out, int64_t value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void append_u64(std::vector<uint8_t>* out, uint64_t value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void set_worker_latency(int32_t module, const std::chrono::steady_clock::time_point& start) {
    if (module >= AI_MODULE_CV && module <= AI_MODULE_YOLO) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        g_last_latency_us[module].store(elapsed, std::memory_order_relaxed);
    }
}

void append_bytes(std::vector<uint8_t>* out, const void* data, int32_t size) {
    append_i32(out, size);
    if (data != nullptr && size > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        out->insert(out->end(), p, p + size);
    }
}

void append_string(std::vector<uint8_t>* out, const std::string& value) {
    append_bytes(out, value.data(), static_cast<int32_t>(value.size()));
}

std::vector<uint8_t> text_payload(const char* text) {
    std::vector<uint8_t> out;
    const char* safe = text == nullptr ? "" : text;
    append_string(&out, safe);
    return out;
}

std::vector<uint8_t> text_payload(const std::string& text) {
    std::vector<uint8_t> out;
    append_string(&out, text);
    return out;
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out) {
    if (path.empty() || out == nullptr) return false;
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) return false;
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0) return false;
    out->assign(static_cast<size_t>(size), 0);
    input.read(reinterpret_cast<char*>(out->data()), size);
    return input.good();
}

std::string json_escape(const char* text) {
    std::ostringstream oss;
    const char* safe = text == nullptr ? "" : text;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(safe); *p != 0; ++p) {
        switch (*p) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (*p < 0x20) {
                    oss << "\\u00";
                    const char* hex = "0123456789abcdef";
                    oss << hex[(*p >> 4) & 0x0f] << hex[*p & 0x0f];
                } else {
                    oss << static_cast<char>(*p);
                }
                break;
        }
    }
    return oss.str();
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

std::string runtime_status_fields(const ai::RuntimeStatus& runtime) {
    return std::string("\"runtime_flavor\":\"") + json_escape(runtime.runtime_flavor.c_str()) +
        "\",\"ort_version\":\"" + json_escape(runtime.ort_version.c_str()) +
        "\",\"ort_path\":\"" + json_escape(runtime.ort_path.c_str()) +
        "\",\"available_providers\":" + providers_json(runtime.available_providers) +
        ",\"requested\":\"" + json_escape(runtime.requested.c_str()) +
        "\",\"active\":\"" + json_escape(runtime.active.c_str()) +
        "\",\"degraded\":" + (runtime.degraded ? "true" : "false") +
        ",\"mixed_cpu_fallback\":" +
            (runtime.mixed_cpu_fallback ? "true" : "false") +
        ",\"device_id\":" + std::to_string(runtime.device_id) +
        ",\"adapter_name\":\"" +
            json_escape(runtime.adapter_name.c_str()) +
        "\",\"selection_basis\":\"" +
            json_escape(runtime.selection_basis.c_str()) +
        "\",\"calibration_key\":\"" +
            json_escape(runtime.calibration_key.c_str()) +
        "\",\"cpu_calibration_ms\":" +
            std::to_string(runtime.cpu_calibration_ms) +
        ",\"directml_calibration_ms\":" +
            std::to_string(runtime.directml_calibration_ms) +
        ",\"active_requests\":" + std::to_string(g_active_requests) +
        ",\"live_connections\":" + std::to_string(g_live_connections.load()) +
        ",\"precision\":\"" + runtime.precision + "\"" +
        ",\"tensorrt_version\":\"" + runtime.tensorrt_version + "\"" +
        ",\"cuda_version\":" + std::to_string(runtime.cuda_version) +
        ",\"driver_version\":" + std::to_string(runtime.driver_version) +
        ",\"driver_file_version\":\"" + runtime.driver_file_version + "\"" +
        ",\"driver_binary_sha256\":\"" + runtime.driver_binary_sha256 + "\"" +
        ",\"execution_slots\":" + std::to_string(runtime.execution_slots) +
        ",\"engine_cache_hit\":" + (runtime.engine_cache_hit ? "true" : "false") +
        ",\"cuda_graph\":" + (runtime.cuda_graph ? "true" : "false") +
        ",\"engine_build_us\":" + std::to_string(runtime.engine_build_us) +
        ",\"engine_cache_key\":\"" + runtime.engine_cache_key + "\"" +
        ",\"reason\":\"" + json_escape(runtime.reason.c_str()) + "\"";
}

std::string runtime_probe_json() {
    ai::RuntimeStatus runtime;
    runtime.runtime_flavor = ai_worker::runtime_flavor_name(ai_worker::kBuildFlavor);
    HMODULE ort_module = GetModuleHandleW(L"onnxruntime.dll");
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    const auto get_api_base = ort_module == nullptr
        ? nullptr
        : reinterpret_cast<GetApiBaseFn>(
            GetProcAddress(ort_module, "OrtGetApiBase"));
    const OrtApiBase* api_base =
        get_api_base == nullptr ? nullptr : get_api_base();
    runtime.ort_version =
        api_base == nullptr || api_base->GetVersionString == nullptr
        ? std::string()
        : std::string(api_base->GetVersionString());
    runtime.available_providers = Ort::GetAvailableProviders();
    runtime.ort_path = ai_runtime::loaded_ort_path_utf8();
    runtime.requested = "probe";
    runtime.active = "none";
    runtime.reason = "";
    runtime.degraded = false;
    return std::string("{") + runtime_status_fields(runtime) +
        ",\"worker_protocol\":" + std::to_string(ai_worker::kVersion) +
        ",\"runtime_cache\":\"" +
            json_escape(ai_runtime::runtime_cache_path_utf8().c_str()) +
        "\",\"runtime_bundle_sha256\":\"" +
            json_escape(ai_runtime::runtime_bundle_hash().c_str()) +
        "\",\"directml_cached\":" +
            (ai_runtime::directml_is_cached() ? "true" : "false") +
        "}";
}

std::string format_yolo_json(const std::vector<AIDetectBox>& boxes, int32_t count, float conf) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << '[';
    int32_t written = 0;
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(boxes.size())));
    for (int32_t i = 0; i < safe_count; ++i) {
        const AIDetectBox& box = boxes[static_cast<size_t>(i)];
        if (box.score < conf) continue;
        if (written++ > 0) {
            oss << ',';
        }
        const std::string fallback = std::to_string(box.class_id);
        const char* label = box.label[0] == '\0' ? fallback.c_str() : box.label;
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
            << ",\"label\":\"" << json_escape(label) << "\"}";
    }
    oss << ']';
    return oss.str();
}

int32_t count_yolo_boxes(const std::vector<AIDetectBox>& boxes, int32_t count, float conf) {
    int32_t total = 0;
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(boxes.size())));
    for (int32_t i = 0; i < safe_count; ++i) {
        if (boxes[static_cast<size_t>(i)].score >= conf) {
            ++total;
        }
    }
    return total;
}

const char* runtime_device_name(int32_t device) {
    switch (device) {
        case AI_DEVICE_AUTO: return "auto";
        case AI_DEVICE_DIRECTML: return "directml";
        case AI_DEVICE_CPU: return "cpu";
        case AI_DEVICE_TENSORRT: return "tensorrt";
        default: return "invalid";
    }
}

int32_t resolve_yolo_intra_threads(int32_t session_count) {
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned int per_session = std::max(1u, hardware / static_cast<unsigned int>(session_count));
    return static_cast<int32_t>(std::min(4u, per_session));
}

int32_t resolve_ocr_intra_threads(int32_t session_count) {
    const unsigned int hardware =
        std::max(1u, std::thread::hardware_concurrency());
    // OCR performs image preprocessing, DB postprocessing, recognition crop
    // preparation, and pipe serialization outside ORT.  Reserve half of the
    // logical processors for that work so four Session pools do not create
    // sixteen continuously busy ORT workers on hybrid CPUs.
    const unsigned int budget_per_session = std::max(
        1u,
        hardware /
            (2u * static_cast<unsigned int>(
                       std::max(1, session_count))));
    return static_cast<int32_t>(std::min(4u, budget_per_session));
}

uint64_t fnv1a64(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t value = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        value ^= bytes[i];
        value *= 1099511628211ull;
    }
    return value;
}

std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string directml_adapter_signature(int32_t device_id) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(
            __uuidof(IDXGIFactory1),
            reinterpret_cast<void**>(&factory))) ||
        factory == nullptr) {
        return "dxgi-unavailable";
    }
    IDXGIAdapter1* adapter = nullptr;
    const HRESULT enum_result =
        factory->EnumAdapters1(static_cast<UINT>(device_id), &adapter);
    factory->Release();
    if (FAILED(enum_result) || adapter == nullptr) {
        return "adapter-unavailable-" + std::to_string(device_id);
    }
    DXGI_ADAPTER_DESC1 description{};
    LARGE_INTEGER driver_version{};
    const HRESULT desc_result = adapter->GetDesc1(&description);
    const HRESULT driver_result = adapter->CheckInterfaceSupport(
        __uuidof(IDXGIDevice), &driver_version);
    adapter->Release();
    if (FAILED(desc_result)) {
        return "adapter-query-failed-" + std::to_string(device_id);
    }
    std::ostringstream out;
    out << std::hex
        << description.VendorId << '-'
        << description.DeviceId << '-'
        << description.SubSysId << '-'
        << description.Revision << '-'
        << static_cast<uint32_t>(description.AdapterLuid.HighPart) << '-'
        << static_cast<uint32_t>(description.AdapterLuid.LowPart);
    if (SUCCEEDED(driver_result)) {
        out << '-' << static_cast<uint64_t>(driver_version.QuadPart);
    }
    return out.str();
}

std::string processor_signature() {
    char identifier[512]{};
    const DWORD length = GetEnvironmentVariableA(
        "PROCESSOR_IDENTIFIER",
        identifier,
        static_cast<DWORD>(std::size(identifier)));
    std::ostringstream out;
    if (length > 0 && length < std::size(identifier)) {
        out << identifier;
    } else {
        out << "cpu";
    }
    out << '-' << std::thread::hardware_concurrency();
    return out.str();
}

std::string make_calibration_key(
    const char* module,
    const std::vector<uint8_t>& model,
    int32_t input_size,
    int32_t session_count,
    int32_t device_id) {
    const std::string cpu = processor_signature();
    std::ostringstream out;
    out << "v23-cal7-" << module
        << '-' << hex_u64(fnv1a64(model.data(), model.size()))
        << '-' << model.size()
        << '-' << input_size
        << '-' << session_count
        << '-' << hex_u64(fnv1a64(cpu.data(), cpu.size()))
        << '-' << directml_adapter_signature(device_id)
        << '-' << ai_runtime::runtime_bundle_hash();
    return out.str();
}

struct CalibrationRecord {
    std::string active;
    int32_t cpu_threads = 0;
    double cpu_ms = -1.0;
    double directml_ms = -1.0;
};

std::mutex g_calibration_cache_mutex;

std::filesystem::path calibration_cache_path() {
    const std::string runtime_cache =
        ai_runtime::runtime_cache_path_utf8();
    if (runtime_cache.empty()) return {};
    return std::filesystem::u8path(runtime_cache) /
        L"performance-calibration-v23.tsv";
}

bool read_calibration_record(
    const std::string& key,
    CalibrationRecord* record) {
    if (record == nullptr) return false;
    std::lock_guard<std::mutex> lock(g_calibration_cache_mutex);
    std::ifstream input(calibration_cache_path(), std::ios::binary);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string stored_key;
        std::string active;
        std::string cpu_threads;
        std::string cpu_ms;
        std::string dml_ms;
        if (!std::getline(fields, stored_key, '\t') ||
            !std::getline(fields, active, '\t') ||
            !std::getline(fields, cpu_threads, '\t') ||
            !std::getline(fields, cpu_ms, '\t') ||
            !std::getline(fields, dml_ms, '\t') ||
            stored_key != key) {
            continue;
        }
        try {
            record->active = active;
            record->cpu_threads = std::stoi(cpu_threads);
            record->cpu_ms = std::stod(cpu_ms);
            record->directml_ms = std::stod(dml_ms);
            return active == "cpu" || active == "directml";
        } catch (const std::exception&) {
            return false;
        }
    }
    return false;
}

void write_calibration_record(
    const std::string& key,
    const CalibrationRecord& record) {
    std::lock_guard<std::mutex> lock(g_calibration_cache_mutex);
    const std::filesystem::path target = calibration_cache_path();
    if (target.empty()) return;
    std::map<std::string, CalibrationRecord> records;
    {
        std::ifstream input(target, std::ios::binary);
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream fields(line);
            std::string stored_key;
            std::string active;
            std::string cpu_threads;
            std::string cpu_ms;
            std::string dml_ms;
            if (!std::getline(fields, stored_key, '\t') ||
                !std::getline(fields, active, '\t') ||
                !std::getline(fields, cpu_threads, '\t') ||
                !std::getline(fields, cpu_ms, '\t') ||
                !std::getline(fields, dml_ms, '\t')) {
                continue;
            }
            try {
                records[stored_key] = CalibrationRecord{
                    active,
                    std::stoi(cpu_threads),
                    std::stod(cpu_ms),
                    std::stod(dml_ms)};
            } catch (const std::exception&) {
            }
        }
    }
    records[key] = record;
    const std::filesystem::path temporary =
        target.wstring() + L".tmp-" + std::to_wstring(GetCurrentProcessId());
    {
        std::ofstream output(
            temporary, std::ios::binary | std::ios::trunc);
        if (!output) return;
        output.imbue(std::locale::classic());
        output << std::setprecision(17);
        for (const auto& item : records) {
            output << item.first << '\t'
                   << item.second.active << '\t'
                   << item.second.cpu_threads << '\t'
                   << item.second.cpu_ms << '\t'
                   << item.second.directml_ms << '\n';
        }
        output.flush();
        if (!output) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return;
        }
    }
    if (!MoveFileExW(
            temporary.c_str(),
            target.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
    }
}

bool make_yolo_params(int32_t input,int32_t device,int32_t ordinal,int32_t sessions,WorkerYoloParams* p,std::string* error) {
    // The host serialized a complete per-model configuration. Worker startup
    // environment must not override or reject a later caller's settings.
    return ai::yolo_parameters(input,device,ordinal,sessions,p,error,false);
}

ai::Config make_yolo_config(
    const std::string& labels_path,
    const std::string& labels_inline,
    const WorkerYoloParams& options,
    int32_t device) {
    ai::Config config;
    config.set_string("runtime.prefer_gpu", "true");
    config.set_int("runtime.device", device);
    config.set_int("runtime.device_id", options.device_id);
    config.set_int("runtime.session_count", options.session_count);
    config.set_int("runtime.intra_op_threads", options.intra_op_threads);
    config.set_string("yolo.backend", "onnxruntime");
    config.set_int("yolo.input_width", options.input_size);
    config.set_int("yolo.input_height", options.input_size);
    config.set_string("yolo.nms_threshold", std::to_string(kYoloInternalNmsThreshold));
    if (!labels_path.empty()) {
        config.set_string("yolo.labels_path", labels_path);
    }
    if (!labels_inline.empty()) {
        config.set_string("yolo.labels_inline", labels_inline);
    }
    config.set_string("ocr.backend", "null");
    return config;
}

std::shared_ptr<WorkerYoloContext> get_yolo_context(int32_t handle) {
    if (handle <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_yolo_registry_mutex);
    const auto it = g_yolo_contexts.find(handle);
    return it == g_yolo_contexts.end() ? nullptr : it->second;
}

int32_t create_yolo_context(int32_t* out_handle) {
    if (out_handle == nullptr) return AI_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_yolo_registry_mutex);
    if (g_next_yolo_handle <= 0 || g_next_yolo_handle == std::numeric_limits<int32_t>::max()) return AI_ERR_RUNTIME;
    const int32_t handle = g_next_yolo_handle++;
    g_yolo_contexts.emplace(handle, std::make_shared<WorkerYoloContext>());
    *out_handle = handle;
    return AI_OK;
}

double median_sample(std::vector<double> samples) {
    if (samples.empty()) return -1.0;
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) * 0.5
        : samples[middle];
}

std::shared_ptr<WorkerYoloPool> build_yolo_pool(
    const std::shared_ptr<const std::vector<uint8_t>>& model,
    const std::string& labels_path, const std::string& labels_inline,
    const WorkerYoloParams& p, int32_t* status, std::string* error) {
    auto config=make_yolo_config(labels_path,labels_inline,p,p.runtime_device);
    return ai::build_yolo_pool_shared(model,std::move(config),p,status,error);
}

int32_t load_yolo_context(
    int32_t handle,
    const std::shared_ptr<const std::vector<uint8_t>>& model,
    const std::string& labels_path,
    const std::string& labels_inline,
    const WorkerYoloParams& options,
    std::string* error) {
    const auto context = get_yolo_context(handle);
    if (!context) return AI_ERR_INVALID_HANDLE;
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->state == WorkerYoloState::Loaded) return AI_ERR_ALREADY_LOADED;
        if (context->state == WorkerYoloState::Loading) return AI_ERR_BUSY;
        if (context->state == WorkerYoloState::Closing) return AI_ERR_INVALID_HANDLE;
        context->state = WorkerYoloState::Loading;
    }

    int32_t build_status = AI_ERR_CONFIG;
    auto pool = build_yolo_pool(model, labels_path, labels_inline, options, &build_status, error);
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->state == WorkerYoloState::Closing) return AI_ERR_INVALID_HANDLE;
    if (!pool) {
        context->state = WorkerYoloState::Empty;
        return build_status;
    }
    context->pool = std::move(pool);
    context->state = WorkerYoloState::Loaded;
    return AI_OK;
}

std::shared_ptr<WorkerYoloPool> get_loaded_yolo_pool(int32_t handle, int32_t* status) {
    const auto context = get_yolo_context(handle);
    if (!context) {
        if (status != nullptr) *status = AI_ERR_INVALID_HANDLE;
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->state == WorkerYoloState::Loading) {
        if (status != nullptr) *status = AI_ERR_BUSY;
        return nullptr;
    }
    if (context->state != WorkerYoloState::Loaded || !context->pool) {
        if (status != nullptr) *status = AI_ERR_BACKEND_NOT_CONFIGURED;
        return nullptr;
    }
    if (status != nullptr) *status = AI_OK;
    return context->pool;
}

int32_t acquire_yolo_slot(const std::shared_ptr<WorkerYoloPool>& pool, ai::Engine** engine) {
    return pool && engine ? pool->acquire(engine) : AI_ERR_INVALID_ARGUMENT;
}
void release_yolo_slot(const std::shared_ptr<WorkerYoloPool>& pool,int32_t index) {
    if(pool && index>=0)pool->release(index);
}

int32_t release_yolo_context(int32_t handle) {
    std::shared_ptr<WorkerYoloContext> context;
    {
        std::lock_guard<std::mutex> lock(g_yolo_registry_mutex);
        const auto it = g_yolo_contexts.find(handle);
        if (it == g_yolo_contexts.end()) return AI_ERR_INVALID_HANDLE;
        context = it->second;
        g_yolo_contexts.erase(it);
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    context->state = WorkerYoloState::Closing;
    context->pool.reset();
    return AI_OK;
}

void clear_yolo_contexts() {
    std::unordered_map<int32_t, std::shared_ptr<WorkerYoloContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(g_yolo_registry_mutex);
        contexts.swap(g_yolo_contexts);
    }
    for (auto& item : contexts) {
        std::lock_guard<std::mutex> lock(item.second->mutex);
        item.second->state = WorkerYoloState::Closing;
        item.second->pool.reset();
    }
}

int32_t yolo_pool_infer_json(
    int32_t handle,
    const uint8_t* image_bytes, size_t image_size,
    float conf,
    int32_t origin_x,
    int32_t origin_y,
    std::string* json,
    std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    AIImage bmp{};
    if (!ai::bmp24_view(image_bytes, image_size, &bmp)) {
        if (error != nullptr) *error = "invalid 24-bit BMP image";
        return AI_ERR_IMAGE_FORMAT;
    }

    ai::yolo_timing.bmp_us = ai::elapsed_us(start);
    int32_t lookup_status = AI_OK;
    const auto pool = get_loaded_yolo_pool(handle, &lookup_status);
    if (!pool) return lookup_status;
    ai::Engine* engine = nullptr;
    const auto wait_start = ai::YoloClock::now();
    const int32_t slot = acquire_yolo_slot(pool, &engine);
    ai::yolo_timing.wait_us = ai::elapsed_us(wait_start);
    if (slot < 0) {
        if (error != nullptr) *error = "yolo model is not loaded";
        return slot;
    }

    struct Lease {
        std::shared_ptr<WorkerYoloPool> pool; int32_t slot;
        ~Lease() { release_yolo_slot(pool, slot); }
    } lease{pool, slot};
    thread_local std::vector<AIDetectBox> boxes;
    boxes.clear();
    const int32_t status = engine->yolo_detect(bmp, conf, pool->nms_threshold, &boxes);
    set_worker_latency(AI_MODULE_YOLO, start);
    pool->last_latency_us.store(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count(), std::memory_order_relaxed);
    if (status < 0) {
        if (error != nullptr) *error = ai::last_error().empty() ? "yolo detect failed" : ai::last_error();
        return status;
    }
    if (!offset_yolo_boxes(&boxes, origin_x, origin_y)) {
        if (error != nullptr) *error = "coordinate origin causes int32 overflow";
        return AI_ERR_INVALID_ARGUMENT;
    }
    const auto json_start = ai::YoloClock::now();
    if (json != nullptr) {
        *json = format_yolo_json(boxes, status, conf);
    }
    ai::yolo_timing.json_us = ai::elapsed_us(json_start);
    return count_yolo_boxes(boxes, status, conf);
}

ai::Config make_ocr_config(
    const std::string& charset_inline,
    int32_t device,
    bool with_detection,
    int32_t session_count,
    bool embedded_defaults,
    const AIOcrRuntimeOptions* options) {
    ai::Config config;
    config.set_string("runtime.prefer_gpu", "true");
    config.set_int("runtime.device", device);
    config.set_int("runtime.thread_count", 0);
    config.set_int("runtime.session_count", std::max<int32_t>(1, session_count));
    const int32_t requested_threads =
        options == nullptr ? 0 : options->intra_op_threads;
    config.set_int(
        "runtime.intra_op_threads",
        requested_threads > 0
            ? requested_threads
            : resolve_ocr_intra_threads(session_count));
    config.set_string("yolo.backend", "null");
    config.set_string("ocr.backend", "onnxruntime");
    config.set_string("ocr.rec_only", with_detection ? "false" : "true");
    config.set_int("ocr.input_height", 48);
    config.set_int("ocr.input_width", 320);
    config.set_string("ocr.channel_order", "bgr");
    config.set_string("ocr.charset_inline", charset_inline);
    if (embedded_defaults) {
        config.set_int("ocr.det_input_width", 0);
        config.set_int("ocr.det_input_height", 0);
        config.set_string("ocr.det_binary_threshold", "0.2");
        config.set_string("ocr.det_box_score_threshold", "0.4");
        config.set_string("ocr.det_unclip_ratio", "1.4");
        config.set_int("ocr.det_min_area", 10);
    }
    if (options != nullptr) {
        if (options->det_input_width > 0) config.set_int("ocr.det_input_width", options->det_input_width);
        if (options->det_input_height > 0) config.set_int("ocr.det_input_height", options->det_input_height);
        if (options->det_binary_threshold > 0.0f) config.set_string("ocr.det_binary_threshold", std::to_string(options->det_binary_threshold));
        if (options->det_box_score_threshold > 0.0f) config.set_string("ocr.det_box_score_threshold", std::to_string(options->det_box_score_threshold));
        if (options->det_unclip_ratio > 0.0f) config.set_string("ocr.det_unclip_ratio", std::to_string(options->det_unclip_ratio));
    }
    return config;
}

std::shared_ptr<OcrPoolContext> create_ocr_pool_for_device(
    const std::vector<uint8_t>& det,
    const std::vector<uint8_t>& rec,
    const std::vector<uint8_t>& keys,
    int32_t device,
    int32_t session_count,
    bool embedded_defaults,
    const AIOcrRuntimeOptions* options,
    int32_t* result_status,
    std::string* error) {
    const int32_t sessions = std::max<int32_t>(1, session_count);
    const std::string charset_inline(
        reinterpret_cast<const char*>(keys.data()), keys.size());
    auto next_pool = std::make_shared<OcrPoolContext>();
    next_pool->slots.reserve(static_cast<size_t>(sessions));
    std::string active_provider;
    std::string candidate_error;
    ai::RuntimeStatus pool_runtime;
    int32_t last_status = AI_ERR_CONFIG;
    for (int32_t i = 0; i < sessions; ++i) {
        auto engine = std::make_unique<ai::Engine>();
        if (!engine->init_ex(nullptr, device, &candidate_error)) {
            last_status = AI_ERR_CONFIG;
            break;
        }
        ai::Config config = make_ocr_config(
            charset_inline,
            device,
            !det.empty(),
            sessions,
            embedded_defaults,
            options);
        const int32_t status =
            engine->ocr_load_models_from_memory_with_config(
                det.empty() ? nullptr : det.data(),
                static_cast<int32_t>(det.size()),
                rec.data(),
                static_cast<int32_t>(rec.size()),
                std::move(config),
                device,
                &candidate_error);
        if (status < 0) {
            last_status = status;
            break;
        }
        const ai::RuntimeStatus slot_status =
            ai::get_thread_runtime_status();
        if (slot_status.active != runtime_device_name(device)) {
            candidate_error =
                std::string("OCR requested ") +
                runtime_device_name(device) +
                " but session activated " + slot_status.active;
            last_status = AI_ERR_RUNTIME;
            break;
        }
        if (active_provider.empty()) {
            active_provider = slot_status.active;
            pool_runtime = slot_status;
        } else if (active_provider != slot_status.active) {
            candidate_error =
                "OCR session pool resolved to mixed execution providers";
            last_status = AI_ERR_RUNTIME;
            break;
        }
        OcrSlot slot;
        slot.engine = std::move(engine);
        next_pool->slots.push_back(std::move(slot));
    }
    if (static_cast<int32_t>(next_pool->slots.size()) != sessions) {
        if (error != nullptr) {
            *error = candidate_error.empty()
                ? std::string(runtime_device_name(device)) +
                      " OCR session pool creation failed"
                : candidate_error;
        }
        if (result_status != nullptr) *result_status = last_status;
        return nullptr;
    }
    pool_runtime.active = runtime_device_name(device);
    pool_runtime.degraded = false;
    next_pool->runtime = std::move(pool_runtime);
    if (result_status != nullptr) *result_status = AI_OK;
    return next_pool;
}

double calibrate_ocr_pool(
    const std::shared_ptr<OcrPoolContext>& pool) {
    if (!pool || pool->slots.empty()) return -1.0;
    constexpr int32_t width = 780;
    constexpr int32_t height = 396;
    constexpr int32_t stride = (width * 3 + 3) & ~3;
    HDC dc = CreateCompatibleDC(nullptr);
    BITMAPINFO bitmap_info{};
    bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap_info.bmiHeader.biWidth = width;
    bitmap_info.bmiHeader.biHeight = -height;
    bitmap_info.bmiHeader.biPlanes = 1;
    bitmap_info.bmiHeader.biBitCount = 24;
    bitmap_info.bmiHeader.biCompression = BI_RGB;
    void* bitmap_pixels = nullptr;
    HBITMAP bitmap = dc == nullptr
        ? nullptr
        : CreateDIBSection(
              dc,
              &bitmap_info,
              DIB_RGB_COLORS,
              &bitmap_pixels,
              nullptr,
              0);
    if (dc == nullptr || bitmap == nullptr ||
        bitmap_pixels == nullptr) {
        if (bitmap != nullptr) DeleteObject(bitmap);
        if (dc != nullptr) DeleteDC(dc);
        return -1.0;
    }
    const HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
    RECT background{0, 0, width, height};
    FillRect(dc, &background, static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH)));
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(0, 0, 0));
    HFONT font = CreateFontW(
        -22,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Microsoft YaHei UI");
    const HGDIOBJ old_font =
        font == nullptr ? nullptr : SelectObject(dc, font);
    static const wchar_t* text_blocks[] = {
        L"最新活动: 12345",
        L"游戏公告 ABCDE",
        L"服务条款 67890",
        L"查看详情 XYZ",
        L"新服为爱追寻",
        L"玩家最高在线",
        L"DirectML TEST",
        L"CPU AUTO MODE",
        L"识别性能校准",
        L"准确度优先",
        L"模型运行状态",
        L"欢迎您的到来",
        L"Session Pool",
        L"文字检测识别",
        L"CQ_AI Benchmark",
        L"退出游戏 End"};
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 2; ++column) {
            const wchar_t* text =
                text_blocks[row * 2 + column];
            TextOutW(
                dc,
                18 + column * 380,
                18 + row * 46,
                text,
                static_cast<int>(std::wcslen(text)));
        }
    }
    GdiFlush();
    AIImage image{
        static_cast<uint8_t*>(bitmap_pixels),
        width,
        height,
        stride,
        AI_IMAGE_BGR24};
    ai::Engine* engine = pool->slots.front().engine.get();
    std::vector<AIOcrLine> lines(128);
    bool valid = true;
    for (int i = 0; i < 2; ++i) {
        if (engine->ocr_recognize(
                image, lines.data(), static_cast<int32_t>(lines.size())) <
            0) {
            valid = false;
            break;
        }
    }
    std::vector<double> samples;
    samples.reserve(7);
    for (int i = 0; valid && i < 7; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (engine->ocr_recognize(
                image, lines.data(), static_cast<int32_t>(lines.size())) <
            0) {
            valid = false;
            break;
        }
        samples.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count());
    }
    if (old_font != nullptr) SelectObject(dc, old_font);
    if (font != nullptr) DeleteObject(font);
    SelectObject(dc, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(dc);
    return valid ? median_sample(std::move(samples)) : -1.0;
}

int32_t tune_ocr_cpu_threads(
    const std::vector<uint8_t>& det,
    const std::vector<uint8_t>& rec,
    const std::vector<uint8_t>& keys,
    int32_t session_count,
    bool embedded_defaults,
    const AIOcrRuntimeOptions* base_options,
    std::string* calibration_key_out,
    double* best_ms_out) {
    const std::string module_key =
        "ocr-cpu-threads-" +
        hex_u64(fnv1a64(det.data(), det.size()));
    const std::string key = make_calibration_key(
        module_key.c_str(),
        rec,
        0,
        session_count,
        0);
    if (calibration_key_out != nullptr) {
        *calibration_key_out = key;
    }
    CalibrationRecord cached;
    if (read_calibration_record(key, &cached) &&
        cached.cpu_threads > 0) {
        if (best_ms_out != nullptr) *best_ms_out = cached.cpu_ms;
        return cached.cpu_threads;
    }
    const unsigned int logical =
        std::max(1u, std::thread::hardware_concurrency());
    const unsigned int per_session = std::max(
        1u,
        logical /
            static_cast<unsigned int>(
                std::max(1, session_count)));
    const int32_t candidates[] = {1, 2, 4, 6, 8};
    int32_t best_threads = 1;
    double best_ms = std::numeric_limits<double>::infinity();
    for (const int32_t threads : candidates) {
        if (static_cast<unsigned int>(threads) > per_session &&
            threads != 1) {
            continue;
        }
        AIOcrRuntimeOptions calibration_options =
            base_options == nullptr
            ? AIOcrRuntimeOptions{}
            : *base_options;
        calibration_options.intra_op_threads = threads;
        int32_t status = AI_ERR_CONFIG;
        std::string ignored_error;
        auto pool = create_ocr_pool_for_device(
            det,
            rec,
            keys,
            AI_DEVICE_CPU,
            1,
            embedded_defaults,
            &calibration_options,
            &status,
            &ignored_error);
        const double elapsed = calibrate_ocr_pool(pool);
        if (elapsed >= 0.0 && elapsed < best_ms) {
            best_ms = elapsed;
            best_threads = threads;
        }
    }
    if (!std::isfinite(best_ms)) best_ms = -1.0;
    write_calibration_record(
        key,
        CalibrationRecord{
            "cpu", best_threads, best_ms, -1.0});
    if (best_ms_out != nullptr) *best_ms_out = best_ms;
    return best_threads;
}

void activate_ocr_pool(
    std::shared_ptr<OcrPoolContext> pool,
    ai::RuntimeStatus runtime) {
    pool->runtime = runtime;
    {
        std::lock_guard<std::mutex> lock(g_ocr_registry_mutex);
        g_ocr_pool = std::move(pool);
    }
    ai::set_runtime_status(std::move(runtime));
}

int32_t load_ocr_pool_from_memory(
    const std::vector<uint8_t>& det,
    const std::vector<uint8_t>& rec,
    const std::vector<uint8_t>& keys,
    int32_t device,
    int32_t session_count,
    bool embedded_defaults,
    const AIOcrRuntimeOptions* options,
    std::string* error) {
    if (rec.empty() || keys.empty()) {
        if (error != nullptr) {
            *error =
                "OCR recognition model and charset are required";
        }
        return AI_ERR_INVALID_ARGUMENT;
    }
    const int32_t sessions =
        std::max<int32_t>(1, session_count);
    AIOcrRuntimeOptions effective_options =
        options == nullptr ? AIOcrRuntimeOptions{} : *options;
    const AIOcrRuntimeOptions* effective_options_ptr = options;
    if (device != AI_DEVICE_AUTO) {
        int32_t status = AI_ERR_CONFIG;
        auto pool = create_ocr_pool_for_device(
            det,
            rec,
            keys,
            device,
            sessions,
            embedded_defaults,
            effective_options_ptr,
            &status,
            error);
        if (!pool) return status;
        ai::RuntimeStatus runtime = pool->runtime;
        runtime.requested = runtime_device_name(device);
        runtime.selection_basis = "explicit";
        activate_ocr_pool(std::move(pool), std::move(runtime));
        return AI_OK;
    }

    const std::string module_key =
        "ocr-" + hex_u64(fnv1a64(det.data(), det.size()));
    const std::string calibration_key = make_calibration_key(
        module_key.c_str(),
        rec,
        0,
        sessions,
        0);
    CalibrationRecord cached;
    if (read_calibration_record(calibration_key, &cached)) {
        const int32_t cached_device =
            cached.active == "directml"
            ? AI_DEVICE_DIRECTML
            : AI_DEVICE_CPU;
        int32_t cached_status = AI_ERR_CONFIG;
        auto cached_pool = create_ocr_pool_for_device(
            det,
            rec,
            keys,
            cached_device,
            sessions,
            embedded_defaults,
            effective_options_ptr,
            &cached_status,
            error);
        if (cached_pool) {
            ai::RuntimeStatus runtime = cached_pool->runtime;
            runtime.requested = "auto";
            runtime.selection_basis =
                "cached_short_benchmark_10_percent_gate";
            runtime.calibration_key = calibration_key;
            runtime.cpu_calibration_ms = cached.cpu_ms;
            runtime.directml_calibration_ms =
                cached.directml_ms;
            runtime.reason =
                "AUTO reused the hardware and model calibration cache";
            activate_ocr_pool(
                std::move(cached_pool), std::move(runtime));
            return AI_OK;
        }
    }

    int32_t dml_status = AI_ERR_CONFIG;
    int32_t cpu_status = AI_ERR_CONFIG;
    std::string dml_error;
    std::string cpu_error;
    auto cpu_pool = create_ocr_pool_for_device(
        det,
        rec,
        keys,
        AI_DEVICE_CPU,
        1,
        embedded_defaults,
        effective_options_ptr,
        &cpu_status,
        &cpu_error);
    const double cpu_ms = calibrate_ocr_pool(cpu_pool);
    auto dml_pool = create_ocr_pool_for_device(
        det,
        rec,
        keys,
        AI_DEVICE_DIRECTML,
        1,
        embedded_defaults,
        effective_options_ptr,
        &dml_status,
        &dml_error);
    const double dml_ms = calibrate_ocr_pool(dml_pool);
    int32_t selected = AI_DEVICE_CPU;
    if (dml_ms >= 0.0 &&
        (cpu_ms < 0.0 || dml_ms <= cpu_ms * 0.90)) {
        selected = AI_DEVICE_DIRECTML;
    }
    if (selected == AI_DEVICE_CPU && cpu_ms < 0.0) {
        if (error != nullptr) {
            *error = "AUTO OCR calibration failed; DirectML: " +
                (dml_error.empty() ? std::string("benchmark failed")
                                   : dml_error) +
                "; CPU: " +
                (cpu_error.empty() ? std::string("benchmark failed")
                                   : cpu_error);
        }
        return cpu_status < 0 ? cpu_status : AI_ERR_RUNTIME;
    }

    int32_t selected_status = AI_ERR_CONFIG;
    std::shared_ptr<OcrPoolContext> selected_pool;
    if (sessions == 1) {
        selected_pool = selected == AI_DEVICE_DIRECTML
            ? std::move(dml_pool)
            : std::move(cpu_pool);
        selected_status = AI_OK;
    } else {
        selected_pool = create_ocr_pool_for_device(
            det,
            rec,
            keys,
            selected,
            sessions,
            embedded_defaults,
            effective_options_ptr,
            &selected_status,
            error);
    }
    if (!selected_pool && selected == AI_DEVICE_DIRECTML &&
        cpu_ms >= 0.0) {
        selected = AI_DEVICE_CPU;
        selected_pool = create_ocr_pool_for_device(
            det,
            rec,
            keys,
            selected,
            sessions,
            embedded_defaults,
            effective_options_ptr,
            &selected_status,
            error);
    }
    if (!selected_pool) return selected_status;
    ai::RuntimeStatus runtime = selected_pool->runtime;
    runtime.requested = "auto";
    runtime.selection_basis =
        "short_benchmark_10_percent_gate";
    runtime.calibration_key = calibration_key;
    runtime.cpu_calibration_ms = cpu_ms;
    runtime.directml_calibration_ms = dml_ms;
    runtime.degraded =
        dml_ms < 0.0 && selected == AI_DEVICE_CPU;
    if (selected == AI_DEVICE_CPU) {
        runtime.reason = dml_ms < 0.0
            ? "AUTO selected CPU because DirectML calibration failed: " +
                  (dml_error.empty() ? std::string("unknown error")
                                     : dml_error)
            : "AUTO selected CPU because DirectML was not at least 10% faster";
    } else {
        runtime.reason =
            "AUTO selected DirectML because calibration exceeded the 10% gate";
    }
    write_calibration_record(
        calibration_key,
        CalibrationRecord{
            runtime_device_name(selected),
            effective_options.intra_op_threads,
            cpu_ms,
            dml_ms});
    activate_ocr_pool(std::move(selected_pool), std::move(runtime));
    return AI_OK;
}

std::shared_ptr<OcrPoolContext> current_ocr_pool() {
    std::lock_guard<std::mutex> lock(g_ocr_registry_mutex);
    return g_ocr_pool;
}

void clear_ocr_pool() {
    std::lock_guard<std::mutex> lock(g_ocr_registry_mutex);
    g_ocr_pool.reset();
}

uint64_t process_private_usage_bytes() {
    using GetProcessMemoryInfoFn = BOOL(WINAPI*)(
        HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    static const auto get_process_memory_info =
        reinterpret_cast<GetProcessMemoryInfoFn>(GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"),
            "K32GetProcessMemoryInfo"));
    if (get_process_memory_info == nullptr) return 0;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!get_process_memory_info(
            GetCurrentProcess(),
            reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),
            sizeof(counters))) {
        return 0;
    }
    return static_cast<uint64_t>(counters.PrivateUsage);
}

void request_ocr_worker_rotation_if_needed() {
    const auto pool = current_ocr_pool();
    if (!pool || pool->runtime.active != "directml") return;
    const int64_t request_count =
        g_directml_ocr_request_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t private_usage = process_private_usage_bytes();
    uint64_t private_baseline =
        g_directml_ocr_private_baseline.load(std::memory_order_relaxed);
    if (private_usage > 0 && private_baseline == 0) {
        g_directml_ocr_private_baseline.compare_exchange_strong(
            private_baseline,
            private_usage,
            std::memory_order_relaxed);
        private_baseline = g_directml_ocr_private_baseline.load(
            std::memory_order_relaxed);
    }
    const bool private_growth_exceeded = private_usage > 0 &&
        private_baseline > 0 && private_usage >= private_baseline &&
        private_usage - private_baseline >=
            kDirectmlOcrPrivateGrowthLimit;
    if (request_count < kDirectmlOcrRequestsPerWorker &&
        !private_growth_exceeded) {
        return;
    }

    // Do not invalidate live YOLO remote handles. OCR-only Workers can be
    // replaced transparently by the x86 proxy, which replays the last
    // successful OCR load request in the new process.
    {
        std::lock_guard<std::mutex> lock(g_yolo_registry_mutex);
        if (!g_yolo_contexts.empty()) return;
    }
    g_ocr_rotation_requested.store(true, std::memory_order_release);
}

void rotate_worker_after_response_if_requested() {
    if (!g_ocr_rotation_requested.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_client_mutex);
        if (g_shutting_down) return;
        g_shutting_down = true;
    }

    std::unique_lock<std::mutex> lock(g_client_mutex);
    const bool drained = g_client_cv.wait_for(
        lock,
        std::chrono::milliseconds(kShutdownDrainMs),
        [] { return g_active_requests <= 1; });
    if (!drained) {
        g_shutting_down = false;
        g_ocr_rotation_requested.store(true, std::memory_order_release);
        return;
    }
    lock.unlock();
    clear_yolo_contexts();
    clear_ocr_pool();
    ExitProcess(0);
}

void worker_client_watchdog() {
    for (;;) {
        Sleep(kClientWatchdogPollMs);

        bool initiate_shutdown = false;
        {
            std::lock_guard<std::mutex> lock(g_client_mutex);
            for (auto it = g_client_processes.begin(); it != g_client_processes.end();) {
                const DWORD state = WaitForSingleObject(it->second, 0);
                if (state == WAIT_OBJECT_0 || state == WAIT_FAILED) {
                    CloseHandle(it->second);
                    it = g_client_processes.erase(it);
                } else {
                    ++it;
                }
            }

            if (g_client_tracking_enabled && g_client_processes.empty() && !g_shutting_down) {
                const auto now = std::chrono::steady_clock::now();
                if (g_no_client_since == std::chrono::steady_clock::time_point{}) {
                    g_no_client_since = now;
                } else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - g_no_client_since).count() >=
                           kOrphanGraceMs) {
                    g_shutting_down = true;
                    initiate_shutdown = true;
                }
            }
        }

        if (!initiate_shutdown) continue;

        clear_yolo_contexts();
        clear_ocr_pool();
        std::unique_lock<std::mutex> lock(g_client_mutex);
        const bool drained = g_client_cv.wait_for(
            lock,
            std::chrono::milliseconds(kShutdownDrainMs),
            [] { return g_active_requests == 0 && g_client_processes.empty(); });
        if (drained) {
            lock.unlock();
            ExitProcess(0);
        }
        g_shutting_down = false;
        g_no_client_since = std::chrono::steady_clock::now();
    }
}

std::string format_ocr_text(const std::vector<AIOcrLine>& lines, int32_t count, int32_t output_format) {
    std::ostringstream out;
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(lines.size())));
    if (output_format == AI_OCR_OUTPUT_JSON) {
        if (safe_count == 0) return "[]";
        out << "{\"lines\":[";
        for (int32_t i = 0; i < safe_count; ++i) {
            if (i > 0) out << ',';
            const AIOcrLine& line = lines[static_cast<size_t>(i)];
            out << "{\"text\":\"" << json_escape(line.text) << "\",\"confidence\":" << line.confidence
                << ",\"box\":{\"x\":" << line.box.x << ",\"y\":" << line.box.y
                << ",\"w\":" << line.box.w << ",\"h\":" << line.box.h << "}}";
        }
        out << "]}";
        return out.str();
    }
    for (int32_t i = 0; i < safe_count; ++i) out << lines[static_cast<size_t>(i)].text;
    return out.str();
}

struct OcrTargetMatch {
    int32_t target_index = 0;
    std::string target;
    AIOcrLine line{};
};

std::string format_ocr_find_multi_text(const std::vector<OcrTargetMatch>& matches) {
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

int32_t ocr_pool_recognize(
    const std::vector<uint8_t>& image_bytes,
    int32_t output_format,
    float min_confidence,
    const std::string& color_filter,
    std::vector<AIOcrLine>* lines,
    std::string* error,
    const ai::OcrTargetGroups* target_groups = nullptr,
    ai::OcrPipelineSelection* selection = nullptr) {
    if (lines == nullptr || (output_format != AI_OCR_OUTPUT_TEXT && output_format != AI_OCR_OUTPUT_JSON) || min_confidence < 0.0f || min_confidence > 1.0f) return AI_ERR_INVALID_ARGUMENT;
    const auto start = std::chrono::steady_clock::now();
    OwnedBmp bmp;
    if (!parse_bmp24(image_bytes.data(), static_cast<int32_t>(image_bytes.size()), &bmp)) {
        if (error != nullptr) *error = "invalid 24-bit BMP image";
        return AI_ERR_IMAGE_FORMAT;
    }
    ai::OcrColorFilter parsed_filter;
    std::string filter_error;
    if (!ai::parse_ocr_color_filter(color_filter.c_str(), &parsed_filter, &filter_error)) {
        if (error != nullptr) *error = filter_error;
        return AI_ERR_INVALID_ARGUMENT;
    }
    OcrLease lease;
    const int32_t acquire_status = lease.acquire(current_ocr_pool());
    if (acquire_status < 0) return acquire_status;
    ai::Engine* engine = lease.engine();
    const int32_t status = ai::run_ocr_candidate_pipeline(
        bmp.image,
        parsed_filter,
        min_confidence,
        target_groups,
        [engine](
            const AIImage& candidate,
            std::vector<AIOcrLine>* attempt,
            ai::OcrRecognitionDiagnostics* diagnostics,
            std::string* attempt_error) {
            int32_t capacity = 16;
            int32_t recognize_status = AI_ERR_RUNTIME;
            for (;;) {
                attempt->assign(static_cast<size_t>(capacity), AIOcrLine{});
                recognize_status = engine->ocr_recognize(
                    candidate, attempt->data(), capacity, diagnostics);
                if (recognize_status < 0) break;
                if (recognize_status > capacity) {
                    recognize_status = AI_ERR_RUNTIME;
                    break;
                }
                attempt->resize(static_cast<size_t>(recognize_status));
                if (recognize_status < capacity) break;
                if (capacity > std::numeric_limits<int32_t>::max() / 2) {
                    recognize_status = AI_ERR_RUNTIME;
                    break;
                }
                capacity *= 2;
            }
            if (recognize_status < 0 && attempt_error != nullptr) {
                const std::string detail = ai::last_error();
                *attempt_error = detail.empty() ? "OCR recognition failed" : detail;
            }
            return recognize_status;
        },
        lines,
        error,
        selection);
    for (int32_t stage = AI_OCR_STAGE_DETECTION; stage <= AI_OCR_STAGE_POSTPROCESS; ++stage) {
        g_last_ocr_stage_us[stage].store(engine->get_ocr_stage_latency_us(stage), std::memory_order_relaxed);
    }
    set_worker_latency(AI_MODULE_OCR, start);
    if (status < 0 && error != nullptr) {
        const std::string detail = ai::last_error();
        *error = detail.empty() ? "OCR recognition failed" : detail;
    }
    if (status < 0) return status;
    return static_cast<int32_t>(lines->size());
}

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
            AIOcrLine match = line;
            if (!preserve_line_box) {
                match.box = approximate_text_box(line, line_text, pos, target);
            }
            std::strncpy(match.text, target.c_str(), AIENGINE_MAX_TEXT - 1);
            match.text[AIENGINE_MAX_TEXT - 1] = '\0';
            output->push_back(match);
            pos = line_text.find(target, pos + std::max<size_t>(1, target.size()));
        }
    }
    if (output->size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return AI_ERR_RUNTIME;
    return static_cast<int32_t>(output->size());
}

void fill_text_result(const AIOcrLine& line, OCRTextResult* out) {
    out->x = line.box.x;
    out->y = line.box.y;
    out->w = line.box.w;
    out->h = line.box.h;
    out->cx = line.box.x + line.box.w / 2;
    out->cy = line.box.y + line.box.h / 2;
    out->score = line.confidence;
}

int32_t ocr_pool_find(const std::vector<uint8_t>& image_bytes, const std::string& target, float min_confidence, std::vector<AIOcrLine>* output, std::string* error) {
    std::vector<AIOcrLine> lines;
    const ai::OcrTargetGroups targets{{target}};
    ai::OcrPipelineSelection selection;
    const int32_t status = ocr_pool_recognize(
        image_bytes,
        AI_OCR_OUTPUT_TEXT,
        min_confidence,
        std::string(),
        &lines,
        error,
        &targets,
        &selection);
    if (status < 0) return status;
    return find_ocr_text_in_lines(
        lines, target, output, selection.used_merged_line_recognition);
}

bool send_response(HANDLE pipe, int32_t status, const std::vector<uint8_t>& payload) {
    if (payload.size() > ai_worker::kMaxPayload) return false;
    ai_worker::ResponseHeader header{ai_worker::kMagic, ai_worker::kVersion, status,
        static_cast<uint32_t>(payload.size()), ai::yolo_timing.request_id, ai::yolo_timing};
    return ai_worker::write_frame(pipe, &header, sizeof(header)) &&
        ai_worker::write_frame(pipe, payload.data(), static_cast<DWORD>(payload.size()));
}

void handle_client(HANDLE pipe) {
    register_client_process(pipe);
    struct ConnectionGuard {
        ConnectionGuard() { ++g_live_connections; }
        ~ConnectionGuard() { --g_live_connections; }
    } connection_guard;
    std::vector<uint8_t> payload;
    for (;;) {
    ai_worker::Header header{};
    // An idle persistent connection is not an executing request. Shutdown,
    // orphan cleanup and OCR rotation drain requests, not connections.
    if (!ai_worker::read_frame(pipe, &header, sizeof(header))) break;
    ai::yolo_timing = {}; ai::yolo_timing.request_id = header.request_id;
    if (header.magic != ai_worker::kMagic || header.version != ai_worker::kVersion ||
        header.payload_size > ai_worker::kMaxPayload) {
        send_response(pipe, AI_ERR_INVALID_ARGUMENT, text_payload("invalid v26 frame")); break;
    }
    ClientGuard client_guard;
    if (!client_guard.active()) {
        send_response(pipe, AI_ERR_BUSY, text_payload("Worker is shutting down")); break;
    }
    try {
    payload.resize(header.payload_size);
    if (!ai_worker::read_frame(pipe, payload.data(), header.payload_size)) break;
    const auto request_start = ai::YoloClock::now();
    if (!g_runtime_ready.load(std::memory_order_acquire) && header.command != ai_worker::CMD_SHUTDOWN) {
        if (!send_response(pipe, AI_ERR_RUNTIME, text_payload(g_runtime_initialization_error))) break;
        continue;
    }

    Reader r(payload);
    std::vector<uint8_t> response;
    int32_t status = AI_ERR_RUNTIME;
    std::string pool_error;

    const auto read_yolo_identity = [&](int32_t* handle) {
        uint64_t instance = 0;
        return r.read_u64(&instance) && r.read_i32(handle) && instance == g_worker_instance_id;
    };
    const auto read_yolo_options = [&](WorkerYoloParams* options) {
        int32_t input_size = 0;
        int32_t runtime_device = 0;
        int32_t device_id = 0;
        int32_t session_count = 0;
        if (options == nullptr || !r.read_i32(&input_size) || !r.read_i32(&runtime_device) ||
            !r.read_i32(&device_id) || !r.read_i32(&session_count)) {
            return false;
        }
        if (!make_yolo_params(input_size, runtime_device, device_id, session_count, options, &pool_error)) return false;
        if (!r.read_i32(&options->intra_op_threads) || !r.read_i32(&options->fp16) ||
            !r.read_i32(&options->graph) || !r.read_i32(&options->validating) ||
            !r.read_string(&options->calibration_file) || !r.read_string(&options->workload_id) ||
            !r.read_string(&options->engine_cache)) return false;
        return options->intra_op_threads >= 0 && options->intra_op_threads <= 8 &&
            (runtime_device==AI_DEVICE_DIRECTML ||runtime_device==AI_DEVICE_TENSORRT ||
                options->intra_op_threads <= 1 || options->intra_op_threads <= ai::physical_cores()/session_count) &&
            options->fp16 >= 0 && options->fp16 <= 1 && options->graph >= 0 && options->graph <= 1 &&
            options->validating >= 0 && options->validating <= 1 &&
            options->calibration_file.size() < 32768 && options->engine_cache.size() < 32768 && options->workload_id.size() <= 256;
    };
    const auto read_string_candidates = [&](std::vector<std::string>* candidates) {
        int32_t count = 0;
        if (candidates == nullptr || !r.read_i32(&count) || count <= 0 || count > 2) return false;
        candidates->clear();
        candidates->reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) {
            std::string value;
            if (!r.read_string(&value) || value.empty()) return false;
            candidates->push_back(std::move(value));
        }
        return true;
    };

    switch (header.command) {
        case ai_worker::CMD_OCR_LOAD_PATH: {
            std::string det, rec, keys;
            int32_t device = 0, sessions = 1;
            if (r.read_string(&det) && r.read_string(&rec) && r.read_string(&keys) && r.read_i32(&device) && r.read_i32(&sessions)) {
                std::vector<uint8_t> det_bytes;
                std::vector<uint8_t> rec_bytes;
                std::vector<uint8_t> key_bytes;
                const bool det_ok = det.empty() || read_file_bytes(det, &det_bytes);
                if (det_ok && read_file_bytes(rec, &rec_bytes) && read_file_bytes(keys, &key_bytes)) {
                    status = load_ocr_pool_from_memory(
                        det_bytes, rec_bytes, key_bytes, device, sessions, false, nullptr, &pool_error);
                } else {
                    pool_error = "failed to read OCR model or charset file";
                    status = AI_ERR_INVALID_ARGUMENT;
                }
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_OCR_LOAD_MEMORY:
        case ai_worker::CMD_OCR_LOAD_MEMORY_EX: {
            std::vector<uint8_t> det, rec, keys;
            int32_t device = 0, sessions = 1;
            AIOcrRuntimeOptions options{};
            const AIOcrRuntimeOptions* options_ptr = nullptr;
            if (r.read_bytes(&det) && r.read_bytes(&rec) && r.read_bytes(&keys) && r.read_i32(&device) && r.read_i32(&sessions)) {
                if (r.remaining() == sizeof(AIOcrRuntimeOptions) &&
                    r.read_i32(&options.det_input_width) && r.read_i32(&options.det_input_height) && r.read_i32(&options.intra_op_threads) &&
                    r.read_f32(&options.det_binary_threshold) && r.read_f32(&options.det_box_score_threshold) && r.read_f32(&options.det_unclip_ratio)) {
                    options_ptr = &options;
                }
                status = load_ocr_pool_from_memory(
                    det, rec, keys, device, sessions,
                    header.command == ai_worker::CMD_OCR_LOAD_MEMORY_EX,
                    options_ptr, &pool_error);
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_OCR_LOAD_EMBEDDED: {
            int32_t device = 0;
            int32_t sessions = 1;
            AIOcrRuntimeOptions options{};
            if (!r.read_i32(&device) || !r.read_i32(&sessions) ||
                !r.read_i32(&options.det_input_width) || !r.read_i32(&options.det_input_height) ||
                !r.read_i32(&options.intra_op_threads) || !r.read_f32(&options.det_binary_threshold) ||
                !r.read_f32(&options.det_box_score_threshold) || !r.read_f32(&options.det_unclip_ratio)) {
                status = AI_ERR_INVALID_ARGUMENT;
                break;
            }
            ai::EmbeddedAsset det_asset;
            ai::EmbeddedAsset rec_asset;
            ai::EmbeddedAsset charset_asset;
            if (!ai::get_embedded_asset(ai::EmbeddedAssetId::OcrDetModel, &det_asset) ||
                !ai::get_embedded_asset(ai::EmbeddedAssetId::OcrRecModel, &rec_asset) ||
                !ai::get_embedded_asset(ai::EmbeddedAssetId::OcrCharset, &charset_asset)) {
                pool_error = std::string(ai_worker::runtime_flavor_name(ai_worker::kBuildFlavor)) +
                    " worker does not contain embedded PP-OCRv6 assets";
                status = AI_ERR_CONFIG;
                break;
            }
            const std::vector<uint8_t> det(
                static_cast<const uint8_t*>(det_asset.data),
                static_cast<const uint8_t*>(det_asset.data) + det_asset.size);
            const std::vector<uint8_t> rec(
                static_cast<const uint8_t*>(rec_asset.data),
                static_cast<const uint8_t*>(rec_asset.data) + rec_asset.size);
            const std::vector<uint8_t> charset(
                static_cast<const uint8_t*>(charset_asset.data),
                static_cast<const uint8_t*>(charset_asset.data) + charset_asset.size);
            status = load_ocr_pool_from_memory(det, rec, charset, device, sessions, true, &options, &pool_error);
            break;
        }
        case ai_worker::CMD_OCR_RECOGNIZE: {
            std::vector<uint8_t> image;
            int32_t fmt = 1;
            int32_t origin_x = 0;
            int32_t origin_y = 0;
            float min_confidence = 0.0f;
            std::string color_filter;
            if (r.read_bytes(&image) && r.read_i32(&fmt) && r.read_f32(&min_confidence) &&
                r.read_string(&color_filter) && r.read_i32(&origin_x) && r.read_i32(&origin_y)) {
                std::vector<AIOcrLine> lines;
                status = ocr_pool_recognize(image, fmt, min_confidence, color_filter, &lines, &pool_error);
                if (status >= 0 && fmt == AI_OCR_OUTPUT_JSON &&
                    !offset_ocr_lines(&lines, origin_x, origin_y)) {
                    pool_error = "coordinate origin causes int32 overflow";
                    status = AI_ERR_INVALID_ARGUMENT;
                }
                if (status >= 0) response = text_payload(format_ocr_text(lines, status, fmt));
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_OCR_FIND_ONE: {
            std::vector<uint8_t> image;
            std::vector<std::string> targets;
            int32_t origin_x = 0;
            int32_t origin_y = 0;
            float min_confidence = 0.0f;
            std::string color_filter;
            if (r.read_bytes(&image) && read_string_candidates(&targets) && r.read_f32(&min_confidence) &&
                r.read_string(&color_filter) && r.read_i32(&origin_x) && r.read_i32(&origin_y)) {
                std::vector<AIOcrLine> recognized_lines;
                const ai::OcrTargetGroups selection_targets{targets};
                ai::OcrPipelineSelection selection;
                status = ocr_pool_recognize(
                    image,
                    AI_OCR_OUTPUT_TEXT,
                    min_confidence,
                    color_filter,
                    &recognized_lines,
                    &pool_error,
                    &selection_targets,
                    &selection);
                std::vector<AIOcrLine> matches;
                for (size_t i = 0; status >= 0 && i < targets.size(); ++i) {
                    status = find_ocr_text_in_lines(
                        recognized_lines,
                        targets[i],
                        &matches,
                        selection.used_merged_line_recognition);
                    if (status > 0) break;
                }
                if (!matches.empty()) {
                    OCRTextResult result{};
                    fill_text_result(matches.front(), &result);
                    if (!offset_ocr_text_result(&result, origin_x, origin_y)) {
                        pool_error = "coordinate origin causes int32 overflow";
                        status = AI_ERR_INVALID_ARGUMENT;
                    } else {
                        status = AI_OK + 1;
                        append_bytes(&response, &result, sizeof(result));
                    }
                }
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_OCR_FIND_MULTI: {
            std::vector<uint8_t> image;
            int32_t target_count = 0;
            int32_t origin_x = 0;
            int32_t origin_y = 0;
            float min_confidence = 0.0f;
            std::string color_filter;
            if (r.read_bytes(&image) && r.read_i32(&target_count) && target_count > 0 && target_count <= 4096) {
                std::vector<OcrTargetMatch> results;
                std::vector<std::vector<std::string>> parsed_targets(static_cast<size_t>(target_count));
                bool valid = true;
                for (std::vector<std::string>& candidates : parsed_targets) {
                    if (!read_string_candidates(&candidates)) {
                        valid = false;
                        break;
                    }
                }
                if (!valid || !r.read_f32(&min_confidence) || !r.read_string(&color_filter) ||
                    !r.read_i32(&origin_x) || !r.read_i32(&origin_y)) {
                    status = AI_ERR_INVALID_ARGUMENT;
                    break;
                }
                std::vector<AIOcrLine> recognized_lines;
                ai::OcrPipelineSelection selection;
                status = ocr_pool_recognize(
                    image,
                    AI_OCR_OUTPUT_TEXT,
                    min_confidence,
                    color_filter,
                    &recognized_lines,
                    &pool_error,
                    &parsed_targets,
                    &selection);
                for (size_t target_index = 0; status >= 0 && target_index < parsed_targets.size(); ++target_index) {
                    std::vector<AIOcrLine> matches;
                    std::string matched_target;
                    for (const std::string& target : parsed_targets[target_index]) {
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
                    if (status < 0) break;
                    for (const AIOcrLine& match : matches) {
                        results.push_back(OcrTargetMatch{static_cast<int32_t>(target_index), matched_target, match});
                    }
                }
                if (status >= 0) {
                    for (OcrTargetMatch& result : results) {
                        if (!offset_ocr_line(&result.line, origin_x, origin_y)) {
                            pool_error = "coordinate origin causes int32 overflow";
                            status = AI_ERR_INVALID_ARGUMENT;
                            break;
                        }
                    }
                    if (status >= 0) {
                        status = static_cast<int32_t>(results.size());
                        response = text_payload(format_ocr_find_multi_text(results));
                    }
                }
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_OCR_FIND_ONE_COORD: {
            std::vector<uint8_t> image;
            std::vector<std::string> targets;
            int32_t origin_x = 0;
            int32_t origin_y = 0;
            float min_confidence = 0.0f;
            std::string color_filter;
            if (r.read_bytes(&image) && read_string_candidates(&targets) && r.read_f32(&min_confidence) &&
                r.read_string(&color_filter) && r.read_i32(&origin_x) && r.read_i32(&origin_y)) {
                std::vector<AIOcrLine> recognized_lines;
                const ai::OcrTargetGroups selection_targets{targets};
                ai::OcrPipelineSelection selection;
                status = ocr_pool_recognize(
                    image,
                    AI_OCR_OUTPUT_TEXT,
                    min_confidence,
                    color_filter,
                    &recognized_lines,
                    &pool_error,
                    &selection_targets,
                    &selection);
                std::vector<AIOcrLine> matches;
                for (size_t i = 0; status >= 0 && i < targets.size(); ++i) {
                    status = find_ocr_text_in_lines(
                        recognized_lines,
                        targets[i],
                        &matches,
                        selection.used_merged_line_recognition);
                    if (status > 0) break;
                }
                if (!matches.empty()) {
                    OCRCoordResult result{matches.front().box.x + matches.front().box.w / 2, matches.front().box.y + matches.front().box.h / 2, matches.front().box.w, matches.front().box.h, 0};
                    if (!offset_ocr_coord_result(&result, origin_x, origin_y)) {
                        pool_error = "coordinate origin causes int32 overflow";
                        status = AI_ERR_INVALID_ARGUMENT;
                    } else {
                        status = AI_OK + 1;
                        append_bytes(&response, &result, sizeof(result));
                    }
                }
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_YOLO_LOAD_PATH: {
            std::string model_path, labels_path;
            int32_t handle = 0;
            WorkerYoloParams options{};
            if (read_yolo_identity(&handle) && r.read_string(&model_path) && r.read_string(&labels_path) && read_yolo_options(&options)) {
                std::vector<uint8_t> model;
                if (read_file_bytes(model_path, &model)) {
                    auto shared_model = std::make_shared<const std::vector<uint8_t>>(std::move(model));
                    status = load_yolo_context(handle, shared_model, labels_path, std::string(), options, &pool_error);
                } else {
                    pool_error = "failed to read YOLO model file: " + model_path;
                    status = AI_ERR_INVALID_ARGUMENT;
                }
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_YOLO_LOAD_MEMORY: {
            std::vector<uint8_t> model, labels;
            int32_t handle = 0;
            WorkerYoloParams options{};
            if (read_yolo_identity(&handle) && r.read_bytes(&model) && r.read_bytes(&labels) && read_yolo_options(&options)) {
                const std::string labels_inline(reinterpret_cast<const char*>(labels.data()), labels.size());
                auto shared_model = std::make_shared<const std::vector<uint8_t>>(std::move(model));
                status = load_yolo_context(handle, shared_model, std::string(), labels_inline, options, &pool_error);
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_YOLO_INFER_JSON: {
            const uint8_t* image = nullptr;
            size_t image_size = 0;
            int32_t handle = 0;
            int32_t origin_x = 0;
            int32_t origin_y = 0;
            float conf = 0.25f;
            const bool identity_ok = read_yolo_identity(&handle);
            if (identity_ok && r.read_view(&image, &image_size) && r.read_f32(&conf) &&
                r.read_i32(&origin_x) && r.read_i32(&origin_y) &&
                std::isfinite(conf) && conf >= 0.0f && conf <= 1.0f) {
                std::string json;
                status = yolo_pool_infer_json(
                    handle, image, image_size, conf, origin_x, origin_y, &json, &pool_error);
                if (status >= 0) response = text_payload(json);
            } else {
                status = identity_ok ? AI_ERR_INVALID_ARGUMENT : AI_ERR_INVALID_HANDLE;
                if (!identity_ok) pool_error = "YOLO handle belongs to an earlier Worker instance; create and load a new handle";
            }
            break;
        }
        case ai_worker::CMD_RELEASE:
            clear_yolo_contexts();
            clear_ocr_pool();
            ai::set_last_error("");
            status = AI_OK;
            break;
        case ai_worker::CMD_OCR_RELEASE:
            clear_ocr_pool();
            status = AI_OK;
            break;
        case ai_worker::CMD_YOLO_RELEASE: {
            int32_t handle = 0;
            status = read_yolo_identity(&handle) ? release_yolo_context(handle) : AI_ERR_INVALID_HANDLE;
            break;
        }
        case ai_worker::CMD_YOLO_CREATE: {
            int32_t handle = 0;
            status = create_yolo_context(&handle);
            if (status >= 0) {
                append_u64(&response, g_worker_instance_id);
                append_i32(&response, handle);
            }
            break;
        }
        case ai_worker::CMD_YOLO_RUNTIME_STATUS: {
            int32_t handle = 0;
            int32_t lookup_status = AI_OK;
            const bool identity_ok = read_yolo_identity(&handle);
            const auto pool = identity_ok ? get_loaded_yolo_pool(handle, &lookup_status) : nullptr;
            if (!pool) {
                status = identity_ok ? lookup_status : AI_ERR_INVALID_HANDLE;
                if (status == AI_OK) status = AI_ERR_INVALID_HANDLE;
                break;
            }
            response = text_payload(std::string("{\"handle\":") + std::to_string(handle) +
                "," + runtime_status_fields(pool->runtime) +
                ",\"device_id\":" + std::to_string(pool->device_id) +
                ",\"session_count\":" + std::to_string(pool->session_count) +
                ",\"intra_op_threads\":" + std::to_string(pool->intra_op_threads) +
                ",\"input_width\":" + std::to_string(pool->input_width) +
                ",\"input_height\":" + std::to_string(pool->input_height) + "}");
            status = AI_OK;
            break;
        }
        case ai_worker::CMD_YOLO_LAST_LATENCY: {
            int32_t handle = 0;
            int32_t lookup_status = AI_OK;
            const bool identity_ok = read_yolo_identity(&handle);
            const auto pool = identity_ok ? get_loaded_yolo_pool(handle, &lookup_status) : nullptr;
            if (pool) {
                append_i64(&response, pool->last_latency_us.load(std::memory_order_relaxed));
                status = AI_OK;
            } else {
                status = identity_ok && lookup_status != AI_OK ? lookup_status : AI_ERR_INVALID_HANDLE;
            }
            break;
        }
        case ai_worker::CMD_RUNTIME_STATUS: {
            const ai::RuntimeStatus runtime = ai::get_runtime_status();
            response = text_payload(std::string("{") + runtime_status_fields(runtime) + "}");
            status = AI_OK;
            break;
        }
        case ai_worker::CMD_GET_LAST_LATENCY: {
            int32_t module = 0;
            if (r.read_i32(&module) && module >= AI_MODULE_CV && module <= AI_MODULE_YOLO) {
                append_i64(&response, g_last_latency_us[module].load(std::memory_order_relaxed));
                status = AI_OK;
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_GET_OCR_STAGE_LATENCY: {
            int32_t stage = 0;
            if (r.read_i32(&stage) && stage >= AI_OCR_STAGE_DETECTION && stage <= AI_OCR_STAGE_POSTPROCESS) {
                append_i64(&response, g_last_ocr_stage_us[stage].load(std::memory_order_relaxed));
                status = AI_OK;
            } else {
                status = AI_ERR_INVALID_ARGUMENT;
            }
            break;
        }
        case ai_worker::CMD_SHUTDOWN:
            {
                std::lock_guard<std::mutex> lock(g_client_mutex);
                g_shutting_down = true;
            }
            {
                std::unique_lock<std::mutex> lock(g_client_mutex);
                const bool drained = g_client_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(kShutdownDrainMs),
                    [] { return g_active_requests <= 1; });
                if (!drained) {
                    g_shutting_down = false;
                    status = AI_ERR_BUSY;
                    response = text_payload(
                        "Worker shutdown is busy: other client requests are still active");
                    break;
                }
            }
            clear_yolo_contexts();
            clear_ocr_pool();
            ai::set_last_error("");
            status = AI_OK;
            send_response(pipe, status, response);
            CloseHandle(pipe);
            ExitProcess(0);
            return;
        default:
            status = AI_ERR_INVALID_ARGUMENT;
            break;
    }

    if (status < 0 && response.empty()) {
        if (!pool_error.empty()) {
            response = text_payload(pool_error);
        } else {
            response = text_payload(ai::last_error());
        }
    }
    if (status >= 0 &&
        (header.command == ai_worker::CMD_OCR_RECOGNIZE ||
         header.command == ai_worker::CMD_OCR_FIND_ONE ||
         header.command == ai_worker::CMD_OCR_FIND_MULTI ||
         header.command == ai_worker::CMD_OCR_FIND_ONE_COORD)) {
        request_ocr_worker_rotation_if_needed();
    }
    ai::yolo_timing.worker_us = ai::elapsed_us(request_start);
    if (!send_response(pipe, status, response)) break;
    if (header.command == ai_worker::CMD_YOLO_INFER_JSON) ai::write_yolo_trace(ai::elapsed_us(request_start), status);
    rotate_worker_after_response_if_requested();
    } catch (const std::exception& e) {
        if (!send_response(pipe, AI_ERR_RUNTIME, text_payload(e.what()))) break;
    } catch (...) {
        if (!send_response(pipe, AI_ERR_RUNTIME, text_payload("unhandled Worker request exception"))) break;
    }
    }
    CloseHandle(pipe);
}

} // namespace

int main(int argc, char** argv) {
    DWORD owner_pid = 0;
    if (argc == 2 && argv != nullptr && argv[1] != nullptr &&
        std::strncmp(argv[1], "--owner-pid=", 12) == 0) {
        try {
            const unsigned long parsed = std::stoul(std::string(argv[1] + 12));
            if (parsed == 0 || parsed > std::numeric_limits<DWORD>::max()) {
                throw std::out_of_range("owner pid");
            }
            owner_pid = static_cast<DWORD>(parsed);
        } catch (...) {
            std::cerr << "{\"error\":\"Invalid --owner-pid argument\"}" << std::endl;
            return 2;
        }
    } else if (argc == 2 && argv != nullptr && argv[1] != nullptr) {
        const std::string command(argv[1]);
        std::string output;
        std::string error;
        if (command == "--third-party-notices") {
            if (ai_runtime::embedded_third_party_notices(&output, &error)) {
                std::cout << output;
                return 0;
            }
        } else if (command == "--verify-embedded-runtime") {
            if (ai_runtime::verify_embedded_runtime(&output, &error)) {
                std::cout << output << std::endl;
                return 0;
            }
        } else if (command == "--runtime-probe") {
            if (ai_runtime::initialize_embedded_runtime(&error) &&
                ai_runtime::ensure_embedded_directml(&error)) {
                try {
                    std::cout << runtime_probe_json() << std::endl;
                    return 0;
                } catch (const std::exception& ex) {
                    error = ex.what();
                }
            }
        } else {
            error = "Unknown worker command: " + command;
        }
        std::cerr << "{\"error\":\"" << json_escape(error.c_str()) << "\"}" << std::endl;
        return 2;
    }
    HANDLE singleton = CreateMutexA(nullptr, TRUE, ai_worker::singleton_name(ai_worker::kBuildFlavor));
    if (singleton == nullptr || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (singleton != nullptr) {
            CloseHandle(singleton);
        }
        return 0;
    }

    if (owner_pid != 0) {
        register_client_process_id(owner_pid);
    }
    std::thread(worker_client_watchdog).detach();

    bool runtime_attempted = false;
    for (;;) {
        HANDLE pipe = CreateNamedPipeA(
            ai_worker::pipe_name(ai_worker::kBuildFlavor),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            1 << 20,
            1 << 20,
            0,
            nullptr);
        if (pipe == INVALID_HANDLE_VALUE) {
            Sleep(1000);
            continue;
        }
        // The first pipe exists before extraction begins, so x86 callers can
        // connect and receive the complete initialization error instead of a
        // generic "worker did not start" result.
        if (!runtime_attempted) {
            runtime_attempted = true;
            std::string runtime_error;
            const bool ready =
                ai_runtime::initialize_embedded_runtime(&runtime_error);
            g_runtime_initialization_error = std::move(runtime_error);
            g_runtime_ready.store(ready, std::memory_order_release);
        }
        const BOOL connected = ConnectNamedPipe(pipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        if (!connected) {
            CloseHandle(pipe);
            continue;
        }
        std::thread(handle_client, pipe).detach();
    }
}
