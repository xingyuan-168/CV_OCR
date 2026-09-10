#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "ai_engine.h"
#include "compact_result_test_utils.h"

#include <assert.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) return {};
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0) return {};
    std::vector<uint8_t> data(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static void print_error(const char* step, int status) {
    const char* error = AI_GetLastError();
    std::cerr << step << " status=" << status << " error=" << error << "\n";
}

static std::string last_error_text() {
    return AI_GetLastError();
}

static std::filesystem::path executable_directory() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    assert(length > 0 && length < path.size());
    path.resize(length);
    return std::filesystem::path(path).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

static int count_substring(const char* value, const char* needle) {
    int count = 0;
    if (value == nullptr || needle == nullptr || needle[0] == '\0') return count;
    const size_t length = std::strlen(needle);
    for (const char* cursor = value; (cursor = std::strstr(cursor, needle)) != nullptr; cursor += length) ++count;
    return count;
}

static float json_float_field(const char* json, const char* field) {
    const std::string marker = std::string("\"") + field + "\":";
    const char* value = std::strstr(json, marker.c_str());
    assert(value != nullptr);
    char* end = nullptr;
    const float parsed = std::strtof(value + marker.size(), &end);
    assert(end != value + marker.size());
    return parsed;
}

static std::string acp_to_utf8(const char* acp) {
#if defined(_WIN32)
    const int acp_size = static_cast<int>(std::strlen(acp));
    const int wide_size = MultiByteToWideChar(CP_ACP, 0, acp, acp_size, nullptr, 0);
    assert(wide_size > 0);
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    assert(MultiByteToWideChar(CP_ACP, 0, acp, acp_size, wide.data(), wide_size) == wide_size);
    const int utf8_size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    assert(utf8_size > 0);
    std::string utf8(static_cast<size_t>(utf8_size), '\0');
    assert(WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_size, utf8.data(), utf8_size, nullptr, nullptr) == utf8_size);
    return utf8;
#else
    return acp;
#endif
}

static std::string wide_to_acp(const std::wstring& value) {
#if defined(_WIN32)
    const int size = WideCharToMultiByte(CP_ACP, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    assert(size > 0);
    std::string acp(static_cast<size_t>(size), '\0');
    assert(WideCharToMultiByte(CP_ACP, 0, value.data(), static_cast<int>(value.size()), acp.data(), size, nullptr, nullptr) == size);
    return acp;
#else
    return std::filesystem::path(value).u8string();
#endif
}

static int32_t infer_json_count(
    int32_t model_handle,
    const std::vector<uint8_t>& image,
    float conf) {
    const char* json = YOLO_InferJson(
        model_handle, image.data(), static_cast<int32_t>(image.size()), conf);
    if (json == nullptr || json[0] == '\0') return -1;
    assert(json[0] == '[' && json[std::strlen(json) - 1] == ']');
    return count_substring(json, "\"class_id\"");
}

int main() {
    assert(std::strcmp(AI_GetVersion(), "CQ_X86/0.14.5") == 0);
    static_assert(sizeof(OCRTextResult) == 28, "OCRTextResult ABI must remain 28 bytes");
    bool directml_available = false;
#if defined(AIENGINE_TEST_EMBED_OCR)
    const AIOcrRuntimeOptions ocr_options{0, 0, 4, 0.0f, 0.0f, 0.0f};
    assert(OCR_LoadEmbeddedModelEx(99, 1, &ocr_options) == AI_ERR_INVALID_ARGUMENT);
    assert(last_error_text().find("0=AUTO") != std::string::npos);
    const int32_t directml_probe = OCR_LoadEmbeddedModelEx(AI_DEVICE_DIRECTML, 1, &ocr_options);
    if (directml_probe == AI_OK) {
        directml_available = true;
        char directml_status[1024]{};
        assert(AI_GetRuntimeStatusJson(directml_status, sizeof(directml_status)) == AI_OK);
        assert(std::strstr(directml_status, "\"requested\":\"directml\"") != nullptr);
        assert(std::strstr(directml_status, "\"active\":\"directml\"") != nullptr);
        assert(OCR_Release() == AI_OK);
    } else {
        const std::string directml_error = last_error_text();
        assert(directml_error.find("DmlExecutionProvider") != std::string::npos ||
               directml_error.find("DirectML") != std::string::npos);
    }
#endif
    const int ocr_load =
#if defined(AIENGINE_TEST_EMBED_OCR)
        OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 1, &ocr_options);
#else
        OCR_LoadModelFromPath("models\\ocr_ppocrv6\\det.fp32.onnx",
            "models\\ocr_ppocrv6\\rec.fp32.onnx",
            "models\\ocr_ppocrv6\\ppocrv6_tiny_dict.txt", AI_DEVICE_CPU, 1);
#endif
    if (ocr_load < 0) {
        print_error("OCR_LoadModelFromPath", ocr_load);
        return 2;
    }

    const std::vector<uint8_t> ocr_image = read_file("examples\\ocr_test_abc123.bmp");
    if (ocr_image.empty()) return 3;
    const char* ocr_json = OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_JSON, 0.0f, nullptr);
    if (ocr_json == nullptr || ocr_json[0] == '\0') {
        print_error("OCR_Recognize", AI_ERR_RUNTIME);
        return 4;
    }
    std::cout << "ocr_json=" << ocr_json << "\n";
    const std::string baseline_ocr_json = ocr_json;
    const float baseline_ocr_x = json_float_field(baseline_ocr_json.c_str(), "x");
    const float baseline_ocr_y = json_float_field(baseline_ocr_json.c_str(), "y");
    const char* shifted_ocr_json = OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_JSON, 0.0f, nullptr, 125, -35);
    assert(shifted_ocr_json != nullptr && shifted_ocr_json[0] != '\0');
    assert(std::fabs(json_float_field(shifted_ocr_json, "x") - (baseline_ocr_x + 125.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_ocr_json, "y") - (baseline_ocr_y - 35.0f)) < 0.01f);
    const std::string baseline_ocr_text = OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_TEXT, 0.0f, nullptr, 0, 0);
    const std::string shifted_ocr_text = OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_TEXT, 0.0f, nullptr, 999, -999);
    assert(shifted_ocr_text == baseline_ocr_text);
    const char* filtered_json = OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_JSON, 0.0f, "FFFFFF-404040");
    assert(filtered_json != nullptr && filtered_json[0] != '\0');
    const char* filtered_multi = OCR_FindMultiText(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        "A|B", 0.0f, "FFFFFF-404040|000000-303030");
    assert(filtered_multi != nullptr);
    std::vector<compact_test::Point> filtered_points;
    assert(compact_test::parse(filtered_multi, &filtered_points) && !filtered_points.empty());
    const char* invalid_filter = OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_JSON, 0.0f, "FFFFFF");
    assert(invalid_filter != nullptr && invalid_filter[0] == '\0');
    assert(std::strstr(AI_GetLastError(), "RRGGBB-RRGGBB") != nullptr);
    const char* ocr_multi = OCR_FindMultiText(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A|B", 0.0f, nullptr);
    assert(ocr_multi != nullptr && ocr_multi[0] != '\0');
    const std::string baseline_ocr_multi = ocr_multi;
    std::vector<compact_test::Point> baseline_multi_points;
    assert(compact_test::parse(baseline_ocr_multi, &baseline_multi_points) &&
        !baseline_multi_points.empty());
    const char* shifted_ocr_multi = OCR_FindMultiText(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        "A|B", 0.0f, nullptr, -120, 330);
    assert(shifted_ocr_multi != nullptr && shifted_ocr_multi[0] != '\0');
    std::vector<compact_test::Point> shifted_multi_points;
    assert(compact_test::parse(shifted_ocr_multi, &shifted_multi_points));
    assert(shifted_multi_points.size() == baseline_multi_points.size());
    for (size_t i = 0; i < baseline_multi_points.size(); ++i) {
        assert(shifted_multi_points[i].id == baseline_multi_points[i].id);
        assert(shifted_multi_points[i].x == baseline_multi_points[i].x - 120);
        assert(shifted_multi_points[i].y == baseline_multi_points[i].y + 330);
    }
    std::vector<std::string> ocr_thread_json(2);
    std::thread ocr_thread_a([&]() {
        ocr_thread_json[0] = OCR_FindMultiText(
            ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A|B", 0.0f, nullptr);
    });
    std::thread ocr_thread_b([&]() {
        ocr_thread_json[1] = OCR_FindMultiText(
            ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "B|A", 0.0f, nullptr);
    });
    ocr_thread_a.join();
    ocr_thread_b.join();
    for (const std::string& result_text : ocr_thread_json) {
        std::vector<compact_test::Point> points;
        assert(compact_test::parse(result_text, &points) && !points.empty());
    }
    assert(AI_GetOcrStageLatencyUs(AI_OCR_STAGE_DETECTION) >= 0);
    assert(AI_GetOcrStageLatencyUs(AI_OCR_STAGE_RECOGNITION) >= 0);
    assert(AI_GetOcrStageLatencyUs(AI_OCR_STAGE_POSTPROCESS) >= 0);
    assert(OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_JSON, -0.01f, nullptr)[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');
    assert(OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_JSON, 1.01f, nullptr)[0] == '\0');
    assert(std::strstr(OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_JSON, 0.99f, nullptr), "\"lines\"") != nullptr);
    assert(std::strcmp(OCR_Recognize(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()),
        AI_OCR_OUTPUT_JSON, 1.0f, nullptr, 50, 60), "[]") == 0);

    OCRTextResult one_text{};
    assert(OCR_FindOneText(
               ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f, &one_text, nullptr) == 1);
    OCRTextResult shifted_text{};
    assert(OCR_FindOneText(
               ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f,
               &shifted_text, nullptr, 400, -250) == 1);
    assert(shifted_text.x == one_text.x + 400 && shifted_text.cx == one_text.cx + 400);
    assert(shifted_text.y == one_text.y - 250 && shifted_text.cy == one_text.cy - 250);
    assert(shifted_text.w == one_text.w && shifted_text.h == one_text.h &&
        shifted_text.score == one_text.score);
    OCRTextResult missing_text{1, 2, 3, 4, 5, 6, 0.5f};
    assert(OCR_FindOneText(
               ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "not-present", 0.0f,
               &missing_text, nullptr, 400, 500) == 0);
    const OCRTextResult zero_text{};
    assert(std::memcmp(&missing_text, &zero_text, sizeof(missing_text)) == 0);
    assert(std::strcmp(OCR_FindMultiText(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "not-present", 0.0f,
        nullptr, 400, 500), "") == 0);
    assert(AI_GetLastError()[0] == '\0');

    OCRCoordResult baseline_coord{};
    OCRCoordResult shifted_coord{};
    assert(OCR_FindOneCoord(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f,
        &baseline_coord, nullptr, 0, 0) == 1);
    assert(OCR_FindOneCoord(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f,
        &shifted_coord, nullptr, -25, 75) == 1);
    assert(shifted_coord.x == baseline_coord.x - 25 && shifted_coord.y == baseline_coord.y + 75);
    assert(shifted_coord.w == baseline_coord.w && shifted_coord.h == baseline_coord.h);
    OCRCoordResult missing_coord{1, 2, 3, 4, 5};
    assert(OCR_FindOneCoord(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "not-present", 0.0f,
        &missing_coord, nullptr, 700, 800) == 0);
    const OCRCoordResult zero_coord{};
    assert(std::memcmp(&missing_coord, &zero_coord, sizeof(missing_coord)) == 0);
    OCRTextResult overflow_text{1, 2, 3, 4, 5, 6, 0.5f};
    assert(OCR_FindOneText(
        ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f,
        &overflow_text, nullptr, std::numeric_limits<int32_t>::max(), 0) ==
        AI_ERR_INVALID_ARGUMENT);
    assert(std::memcmp(&overflow_text, &zero_text, sizeof(overflow_text)) == 0);
    const auto run_find_one_stress = [&](int thread_count, int iterations) {
        std::vector<int32_t> failures(static_cast<size_t>(thread_count), 0);
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(thread_count));
        for (int i = 0; i < thread_count; ++i) {
            workers.emplace_back([&, i]() {
                for (int n = 0; n < iterations; ++n) {
                    OCRTextResult result{};
                    if (OCR_FindOneText(
                            ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f, &result, nullptr) != 1) {
                        ++failures[static_cast<size_t>(i)];
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();
        assert(std::all_of(failures.begin(), failures.end(), [](int32_t value) { return value == 0; }));
    };

    run_find_one_stress(8, 100);
    assert(OCR_Release() == AI_OK);
    assert(OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 4, &ocr_options) == AI_OK);
    run_find_one_stress(16, 100);
    assert(OCR_Release() == AI_OK);

    assert(OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 1, &ocr_options) == AI_OK);
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<int32_t> release_status(16, AI_ERR_RUNTIME);
    std::vector<std::thread> release_threads;
    for (size_t i = 0; i < release_status.size(); ++i) {
        release_threads.emplace_back([&, i]() {
            ++ready;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            OCRTextResult result{};
            release_status[i] = OCR_FindOneText(
                ocr_image.data(), static_cast<int32_t>(ocr_image.size()), "A", 0.0f, &result, nullptr);
        });
    }
    while (ready.load(std::memory_order_acquire) != static_cast<int>(release_threads.size())) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    assert(OCR_Release() == AI_OK);
    for (auto& worker : release_threads) worker.join();
    assert(std::all_of(release_status.begin(), release_status.end(), [](int32_t value) {
        return value == 1 || value == AI_ERR_BACKEND_NOT_CONFIGURED;
    }));

    assert(OCR_Release() == AI_OK);
    assert(OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_JSON, 0.0f, nullptr)[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');

    const std::filesystem::path ocr_relative_dir = executable_directory() / L"CV_小图OCR模型";
    std::filesystem::create_directories(ocr_relative_dir);
    std::filesystem::copy_file(L"models\\ocr_ppocrv6\\det.fp32.onnx", ocr_relative_dir / L"检测.onnx", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(L"models\\ocr_ppocrv6\\rec.fp32.onnx", ocr_relative_dir / L"识别.onnx", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(L"models\\ocr_ppocrv6\\ppocrv6_tiny_dict.txt", ocr_relative_dir / L"字符表.txt", std::filesystem::copy_options::overwrite_existing);
    const std::string ocr_det_acp = wide_to_acp(L"CV_小图OCR模型\\检测.onnx");
    const std::string ocr_rec_acp = wide_to_acp(L"CV_小图OCR模型\\识别.onnx");
    const std::string ocr_keys_acp = wide_to_acp(L"CV_小图OCR模型\\字符表.txt");
    assert(OCR_LoadModelFromPath(ocr_det_acp.c_str(), ocr_rec_acp.c_str(), ocr_keys_acp.c_str(), AI_DEVICE_CPU, 1) == AI_OK);
    assert(OCR_Recognize(ocr_image.data(), static_cast<int32_t>(ocr_image.size()), AI_OCR_OUTPUT_TEXT, 0.0f, nullptr)[0] != '\0');
    assert(OCR_Release() == AI_OK);

    AI_Release();

    constexpr int32_t smc_input_size = 640;
    constexpr int32_t best_input_size = 320;
    constexpr int32_t cpu_device = AI_DEVICE_CPU;
    constexpr int32_t auto_device = AI_DEVICE_AUTO;
    int32_t yolo1 = 0;
    int32_t yolo2 = 0;
    int32_t yolo_memory = 0;
    int32_t yolo_directml = 0;
    assert(YOLO_Create(&yolo1) == AI_OK && yolo1 > 0);
    assert(YOLO_Create(&yolo2) == AI_OK && yolo2 > 0 && yolo2 != yolo1);
    assert(YOLO_Create(&yolo_memory) == AI_OK && yolo_memory > 0 && yolo_memory != yolo1 && yolo_memory != yolo2);
    if (directml_available) {
        assert(YOLO_Create(&yolo_directml) == AI_OK && yolo_directml > 0);
    }
    assert(YOLO_LoadModelFromPath(yolo1, "unused.onnx", "", 123, cpu_device, 0, 2) == AI_ERR_INVALID_ARGUMENT);
    const std::string size_error = last_error_text();
    assert(size_error.find("input_size must be 0, 320, or 640") != std::string::npos);
    assert(YOLO_LoadModelFromPath(yolo1, "unused.onnx", "", 320, 99, 0, 2) == AI_ERR_INVALID_ARGUMENT);
    assert(last_error_text().find("0=AUTO") != std::string::npos);
    const std::string yolo_model_path = std::filesystem::absolute(std::filesystem::u8path("models\\yolo\\smc.onnx")).u8string();
    const std::string yolo_labels_path = std::filesystem::absolute(std::filesystem::u8path("models\\yolo\\smc.txt")).u8string();
    const std::string yolo_model_path_2 = std::filesystem::absolute(std::filesystem::u8path("models\\yolo\\best.onnx")).u8string();
    const std::filesystem::path relative_model_dir = executable_directory() / L"CV_小图YOLO模型";
    const std::filesystem::path relative_model_file = relative_model_dir / L"best.onnx";
    const std::filesystem::path relative_labels_file = relative_model_dir / L"标签.txt";
    std::filesystem::create_directories(relative_model_dir);
    std::filesystem::copy_file(
        std::filesystem::u8path(yolo_model_path_2), relative_model_file,
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(
        std::filesystem::u8path(yolo_labels_path), relative_labels_file,
        std::filesystem::copy_options::overwrite_existing);
    const std::string relative_model_path = wide_to_acp(L"CV_小图YOLO模型\\best.onnx");
    const std::string relative_labels_path = wide_to_acp(L"CV_小图YOLO模型\\标签.txt");
    const std::vector<uint8_t> yolo_image = read_file("tests\\fixtures\\yolo\\1.bmp");
    std::vector<uint8_t> yolo_memory_model = read_file(yolo_model_path_2.c_str());
    if (yolo_image.empty()) return 5;
    if (yolo_memory_model.empty()) return 5;
    const std::string missing_model_path = std::filesystem::absolute("missing-v19-model.onnx").u8string();
    assert(YOLO_LoadModelFromPath(yolo1, missing_model_path.c_str(), "", smc_input_size, cpu_device, 0, 2) == AI_ERR_INVALID_ARGUMENT);
    const std::string missing_model_error = last_error_text();
    assert(missing_model_error.find("YOLO_LoadModelFromPath") != std::string::npos);
    assert(missing_model_error.find("missing-v19-model.onnx") != std::string::npos);
    const uint8_t invalid_onnx[] = {0x08, 0x01, 0x12, 0x00};
    assert(YOLO_LoadModelFromMemory(
        yolo_memory, invalid_onnx, static_cast<int32_t>(sizeof(invalid_onnx)),
        nullptr, 0, best_input_size, cpu_device, 0, 2) < 0);
    const std::string invalid_onnx_error = last_error_text();
    assert(invalid_onnx_error.find("YOLO_LoadModelFromMemory") != std::string::npos);
    assert(invalid_onnx_error.find("runtime error") == std::string::npos);
    assert(YOLO_LoadModelFromPath(yolo1, yolo_model_path.c_str(), yolo_labels_path.c_str(), best_input_size, cpu_device, 0, 2) == AI_ERR_INVALID_ARGUMENT);
    const int yolo_load = YOLO_LoadModelFromPath(yolo1, yolo_model_path.c_str(), yolo_labels_path.c_str(), smc_input_size, cpu_device, 0, 2);
    if (yolo_load < 0) {
        print_error("YOLO_LoadModelFromPath", yolo_load);
        return 5;
    }

    assert(YOLO_LoadModelFromPath(yolo2, relative_model_path.c_str(), relative_labels_path.c_str(), 0, auto_device, 0, 1) == AI_OK);
    if (yolo_directml > 0) {
        assert(YOLO_LoadModelFromPath(
            yolo_directml, relative_model_path.c_str(), relative_labels_path.c_str(),
            0, AI_DEVICE_DIRECTML, 0, 1) == AI_OK);
    }
    assert(YOLO_LoadModelFromMemory(
        yolo_memory, yolo_memory_model.data(), static_cast<int32_t>(yolo_memory_model.size()),
        nullptr, 0, 0, cpu_device, 0, 2) == AI_OK);
    yolo_memory_model.clear();
    yolo_memory_model.shrink_to_fit();
    assert(YOLO_InferJson(yolo_memory, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '[');
    assert(YOLO_LoadModelFromPath(yolo1, yolo_model_path.c_str(), yolo_labels_path.c_str(), smc_input_size, cpu_device, 0, 2) == AI_ERR_ALREADY_LOADED);
    const int yolo_status = infer_json_count(yolo1, yolo_image, 0.25f);
    if (yolo_status < 0) {
        print_error("YOLO_InferJson", yolo_status);
        return 6;
    }
    const char* yolo_json = YOLO_InferJson(yolo1, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f);
    assert(yolo_json != nullptr && yolo_json[0] == '[');
    const std::string baseline_yolo_json = yolo_json;
    const float x1 = json_float_field(baseline_yolo_json.c_str(), "x1");
    const float y1 = json_float_field(baseline_yolo_json.c_str(), "y1");
    const float x2 = json_float_field(baseline_yolo_json.c_str(), "x2");
    const float y2 = json_float_field(baseline_yolo_json.c_str(), "y2");
    const float cx = json_float_field(baseline_yolo_json.c_str(), "cx");
    const float cy = json_float_field(baseline_yolo_json.c_str(), "cy");
    assert(x2 > x1 && y2 > y1);
    assert(std::fabs(cx - (x1 + x2) * 0.5f) < 0.001f);
    assert(std::fabs(cy - (y1 + y2) * 0.5f) < 0.001f);
    const char* shifted_yolo_json = YOLO_InferJson(
        yolo1, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f, 100, -50);
    assert(shifted_yolo_json != nullptr && shifted_yolo_json[0] == '[');
    assert(std::fabs(json_float_field(shifted_yolo_json, "x1") - (x1 + 100.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_yolo_json, "x2") - (x2 + 100.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_yolo_json, "cx") - (cx + 100.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_yolo_json, "y1") - (y1 - 50.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_yolo_json, "y2") - (y2 - 50.0f)) < 0.01f);
    assert(std::fabs(json_float_field(shifted_yolo_json, "cy") - (cy - 50.0f)) < 0.01f);
    assert(std::strcmp(YOLO_InferJson(
        yolo1, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 1.0f, 100, 200), "[]") == 0);
    assert(YOLO_InferJson(
        yolo1, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f,
        std::numeric_limits<int32_t>::max(), 0)[0] == '\0');
    const auto verify_yolo_origin = [&](int32_t handle) {
        const std::string baseline = YOLO_InferJson(
            handle, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.0f, 0, 0);
        assert(!baseline.empty() && baseline.front() == '[' && baseline != "[]");
        const float base_x1 = json_float_field(baseline.c_str(), "x1");
        const float base_x2 = json_float_field(baseline.c_str(), "x2");
        const float base_cx = json_float_field(baseline.c_str(), "cx");
        const float base_y1 = json_float_field(baseline.c_str(), "y1");
        const float base_y2 = json_float_field(baseline.c_str(), "y2");
        const float base_cy = json_float_field(baseline.c_str(), "cy");
        const char* shifted = YOLO_InferJson(
            handle, yolo_image.data(), static_cast<int32_t>(yolo_image.size()),
            0.0f, -333, 444);
        assert(shifted != nullptr && shifted[0] == '[');
        assert(std::fabs(json_float_field(shifted, "x1") - (base_x1 - 333.0f)) < 0.02f);
        assert(std::fabs(json_float_field(shifted, "x2") - (base_x2 - 333.0f)) < 0.02f);
        assert(std::fabs(json_float_field(shifted, "cx") - (base_cx - 333.0f)) < 0.02f);
        assert(std::fabs(json_float_field(shifted, "y1") - (base_y1 + 444.0f)) < 0.02f);
        assert(std::fabs(json_float_field(shifted, "y2") - (base_y2 + 444.0f)) < 0.02f);
        assert(std::fabs(json_float_field(shifted, "cy") - (base_cy + 444.0f)) < 0.02f);
    };
    verify_yolo_origin(yolo2);
    if (yolo_directml > 0) verify_yolo_origin(yolo_directml);
    const std::string yolo_json_utf8 = acp_to_utf8(baseline_yolo_json.c_str());
    assert(yolo_json_utf8.find("\"label\":\"") != std::string::npos);
    assert(std::any_of(yolo_json_utf8.begin(), yolo_json_utf8.end(), [](unsigned char ch) { return ch >= 0x80; }));
    std::cout << "yolo_status=" << yolo_status << " json=" << baseline_yolo_json << "\n";
    char runtime_status[1024]{};
    assert(YOLO_GetRuntimeStatusJson(yolo1, runtime_status, sizeof(runtime_status)) == AI_OK);
    assert(std::strstr(runtime_status, "\"runtime_flavor\":\"core\"") != nullptr);
    assert(std::strstr(runtime_status, "\"ort_version\":\"1.24.4\"") != nullptr);
    assert(std::strstr(runtime_status, "\"available_providers\":[") != nullptr);
    assert(std::strstr(runtime_status, "\"requested\":\"cpu\"") != nullptr);
    assert(std::strstr(runtime_status, "\"active\":\"cpu\"") != nullptr);
    assert(std::strstr(runtime_status, "\"degraded\":false") != nullptr);
    assert(std::strstr(runtime_status, "\"input_width\":640") != nullptr);
    assert(std::strstr(runtime_status, "\"input_height\":640") != nullptr);
    char runtime_status_2[1024]{};
    assert(YOLO_GetRuntimeStatusJson(yolo2, runtime_status_2, sizeof(runtime_status_2)) == AI_OK);
    assert(std::strstr(runtime_status_2, "\"runtime_flavor\":\"core\"") != nullptr);
    assert(std::strstr(runtime_status_2, "\"requested\":\"auto\"") != nullptr);
    const bool auto_directml =
        std::strstr(runtime_status_2, "\"active\":\"directml\"") != nullptr;
    const bool auto_cpu =
        std::strstr(runtime_status_2, "\"active\":\"cpu\"") != nullptr;
    assert(auto_directml || auto_cpu);
    if (auto_directml) {
        assert(std::strstr(runtime_status_2, "\"degraded\":false") != nullptr);
        assert(std::strstr(runtime_status_2, "\"mixed_cpu_fallback\":true") != nullptr);
    } else {
        assert(std::strstr(runtime_status_2, "\"degraded\":true") != nullptr);
        assert(std::strstr(runtime_status_2, "DirectML") != nullptr);
    }
    assert(std::strstr(runtime_status_2, "\"input_width\":320") != nullptr);
    assert(std::strstr(runtime_status_2, "\"input_height\":320") != nullptr);
    assert(YOLO_GetLastLatencyUs(yolo1) >= 0);
    const int32_t full_result_count = infer_json_count(yolo2, yolo_image, 0.0f);
    assert(full_result_count > 256);
    const char* full_json = YOLO_InferJson(
        yolo2, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.0f);
    assert(count_substring(full_json, "\"class_id\"") == full_result_count);
    const int32_t strict_conf = infer_json_count(yolo1, yolo_image, 0.99f);
    assert(strict_conf <= yolo_status);

    std::vector<int> thread_status(6, -999);
    std::vector<std::string> thread_json(thread_status.size());
    std::vector<std::thread> threads;
    for (int i = 0; i < static_cast<int>(thread_status.size()); ++i) {
        threads.emplace_back([&, i]() {
            const int32_t handle = (i % 2 == 0) ? yolo1 : yolo2;
            const char* json = YOLO_InferJson(handle, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f);
            thread_json[static_cast<size_t>(i)] = json == nullptr ? "" : json;
            thread_status[static_cast<size_t>(i)] = infer_json_count(handle, yolo_image, 0.25f);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const int status : thread_status) {
        if (status < 0) {
            print_error("YOLO_InferJson concurrent", status);
            return 7;
        }
    }
    for (const auto& json : thread_json) assert(!json.empty() && json.front() == '[');
    std::cout << "yolo_concurrent_threads=" << thread_status.size() << "\n";
    assert(YOLO_Release(yolo1) == AI_OK);
    assert(YOLO_Release(yolo1) == AI_ERR_INVALID_HANDLE);
    assert(YOLO_InferJson(yolo1, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');
    assert(YOLO_InferJson(yolo2, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '[');
    assert(YOLO_Release(yolo_memory) == AI_OK);
    if (yolo_directml > 0) assert(YOLO_Release(yolo_directml) == AI_OK);
#if defined(_M_IX86)
    assert(AI_ShutdownWorker() == AI_OK);
    assert(YOLO_InferJson(yolo2, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '\0');
#else
    assert(YOLO_Release(yolo2) == AI_OK);
    assert(AI_ShutdownWorker() == AI_OK);
#endif
    int32_t yolo3 = 0;
    const int32_t restart_create = YOLO_Create(&yolo3);
    if (restart_create < 0) {
        print_error("YOLO_Create after restart", restart_create);
        return 8;
    }
    assert(YOLO_LoadModelFromPath(yolo3, yolo_model_path.c_str(), yolo_labels_path.c_str(), 0, auto_device, 0, 1) == AI_OK);
    char runtime_status_3[1024]{};
    assert(YOLO_GetRuntimeStatusJson(yolo3, runtime_status_3, sizeof(runtime_status_3)) == AI_OK);
    assert(std::strstr(runtime_status_3, "\"input_width\":640") != nullptr);
    assert(std::strstr(runtime_status_3, "\"input_height\":640") != nullptr);
    assert(YOLO_InferJson(yolo3, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '[');
    AI_Release();
    assert(YOLO_InferJson(yolo3, yolo_image.data(), static_cast<int32_t>(yolo_image.size()), 0.25f)[0] == '\0');
    assert(AI_ShutdownWorker() == AI_OK);
    std::error_code cleanup_error;
    std::filesystem::remove(relative_model_file, cleanup_error);
    std::filesystem::remove(relative_labels_file, cleanup_error);
    std::filesystem::remove(relative_model_dir, cleanup_error);
    std::filesystem::remove_all(ocr_relative_dir, cleanup_error);
    return 0;
}
