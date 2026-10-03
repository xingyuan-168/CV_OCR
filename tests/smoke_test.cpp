#include "ai_engine.h"

#include <assert.h>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

float json_float_field(const char* json, const char* field) {
    const std::string marker = std::string("\"") + field + "\":";
    const char* value = std::strstr(json, marker.c_str());
    assert(value != nullptr);
    char* end = nullptr;
    const float parsed = std::strtof(value + marker.size(), &end);
    assert(end != value + marker.size());
    return parsed;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN64)
    assert(std::strcmp(AI_GetVersion(), "CQ_AI_x64/0.14.6") == 0);
#else
    assert(std::strcmp(AI_GetVersion(), "CQ_X86/0.14.6") == 0);
#endif
    static_assert(sizeof(OCRTextResult) == 28, "OCRTextResult ABI must remain 28 bytes");
    assert(argc > 0 && argv != nullptr && argv[0] != nullptr);
    const std::filesystem::path config_path =
        std::filesystem::absolute(std::filesystem::u8path(argv[0])).parent_path() / "mock_runtime.ini";
    std::ofstream config(config_path);
    config << "yolo.backend=mock\n";
    config << "ocr.backend=mock\n";
    config << "ocr.mock_text=A你B好\n";
    config.close();

    assert(AI_InitEx(nullptr, AI_DEVICE_CPU) == AI_OK);
    assert(AI_GetLastError() != nullptr && AI_GetLastError()[0] == '\0');

    std::atomic<int> error_threads_ready{0};
    std::string threshold_error;
    std::string color_error;
    std::thread threshold_thread([&]() {
        assert(AI_CvThreshold(nullptr, nullptr, 128, nullptr, 0) == AI_ERR_INVALID_ARGUMENT);
        error_threads_ready.fetch_add(1, std::memory_order_release);
        while (error_threads_ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        threshold_error = AI_GetLastError();
    });
    std::thread color_thread([&]() {
        assert(AI_CvMeanColor(nullptr, nullptr, nullptr) == AI_ERR_INVALID_ARGUMENT);
        error_threads_ready.fetch_add(1, std::memory_order_release);
        while (error_threads_ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        color_error = AI_GetLastError();
    });
    threshold_thread.join();
    color_thread.join();
    assert(threshold_error.find("AI_CvThreshold") != std::string::npos);
    assert(color_error.find("AI_CvMeanColor") != std::string::npos);

    const int width = 8;
    const int height = 8;
    std::vector<uint8_t> pixels(static_cast<size_t>(width * height * 3), 32);
    AIImage image{pixels.data(), width, height, width * 3, AI_IMAGE_BGR24};

    std::vector<uint8_t> template_pixels = pixels;
    AIImage templ{template_pixels.data(), width, height, width * 3, AI_IMAGE_BGR24};
    const char* cv_json = AI_CvFindImages(&image, &templ, 1, 0.0f);
    assert(cv_json != nullptr && cv_json[0] == '[');

    assert(AI_OcrLoadModels(nullptr, nullptr, "mock_runtime.ini", AI_DEVICE_CPU) == AI_OK);
    const char* ocr_json = AI_OcrRecognize(&image, 0.0f);
    assert(ocr_json != nullptr && std::strstr(ocr_json, u8"A你B好") != nullptr);
    const char* find_json = AI_OcrFindText(&image, u8"你B", 0.0f);
    assert(find_json != nullptr && find_json[0] == '[');
    assert(std::strstr(find_json, "\"box\":{\"x\":2,\"y\":0,\"w\":4,\"h\":8}") != nullptr);

#if !defined(_M_IX86)
    int32_t yolo = 0;
    assert(AI_YoloCreate(&yolo) == AI_OK);
    assert(AI_YoloLoadModel(yolo, nullptr, "mock_runtime.ini", 123, AI_DEVICE_CPU, 0, 1) == AI_ERR_INVALID_ARGUMENT);
    const char* options_error = AI_GetLastError();
    assert(std::strstr(options_error, "input_size must be 0, 320, or 640") != nullptr);
    assert(AI_YoloLoadModel(yolo, nullptr, "mock_runtime.ini", 320, AI_DEVICE_CPU, 0, 1) == AI_OK);
    const char* yolo_json = AI_YoloInfer(yolo, &image, 0.0f);
    assert(yolo_json != nullptr && yolo_json[0] == '[');
    const float x1 = json_float_field(yolo_json, "x1");
    const float y1 = json_float_field(yolo_json, "y1");
    const float x2 = json_float_field(yolo_json, "x2");
    const float y2 = json_float_field(yolo_json, "y2");
    const float cx = json_float_field(yolo_json, "cx");
    const float cy = json_float_field(yolo_json, "cy");
    assert(x2 > x1 && y2 > y1);
    assert(std::fabs(cx - (x1 + x2) * 0.5f) < 0.0001f);
    assert(std::fabs(cy - (y1 + y2) * 0.5f) < 0.0001f);
    assert(AI_YoloRelease(yolo) == AI_OK);
#endif

    AI_OcrRelease();
    AI_Release();
    std::cout << "JSON C ABI smoke passed\n";
    return 0;
}
