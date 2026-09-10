#include "ai_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

// 读取完整二进制文件，主要用于加载 BMP 测试图。
std::vector<uint8_t> read_file(const char* path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input.is_open()) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0) {
        return {};
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// 将未压缩 24/32 位 BMP 解码成 AIImage 视图，并用 keepalive 保存原始字节生命周期。
bool image_from_bmp(const char* path, std::vector<uint8_t>* keepalive, AIImage* image) {
    std::vector<uint8_t> data = read_file(path);
    if (data.size() < 54 || data[0] != 'B' || data[1] != 'M') {
        return false;
    }

    const auto read_u16 = [&](size_t offset) -> uint16_t {
        return static_cast<uint16_t>(data[offset] | (data[offset + 1] << 8));
    };
    const auto read_u32 = [&](size_t offset) -> uint32_t {
        return static_cast<uint32_t>(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24));
    };
    const auto read_i32 = [&](size_t offset) -> int32_t {
        return static_cast<int32_t>(read_u32(offset));
    };

    const uint32_t pixel_offset = read_u32(10);
    const int32_t width = read_i32(18);
    const int32_t raw_height = read_i32(22);
    const uint16_t bpp = read_u16(28);
    const uint32_t compression = read_u32(30);
    if (width <= 0 || raw_height == 0 || compression != 0 || (bpp != 24 && bpp != 32)) {
        return false;
    }

    const int32_t height = raw_height < 0 ? -raw_height : raw_height;
    const int32_t channels = bpp / 8;
    const int32_t stride = ((width * channels + 3) / 4) * 4;
    if (pixel_offset + static_cast<uint32_t>(stride * height) > data.size()) {
        return false;
    }

    *keepalive = std::move(data);
    if (raw_height > 0) {
        image->data = keepalive->data() + pixel_offset + static_cast<size_t>((height - 1) * stride);
        image->stride = -stride;
    } else {
        image->data = keepalive->data() + pixel_offset;
        image->stride = stride;
    }
    image->width = width;
    image->height = height;
    image->format = bpp == 24 ? AI_IMAGE_BGR24 : AI_IMAGE_BGRA32;
    return true;
}

// 打印 DLL 最近错误，便于区分模型加载失败和测试图读取失败。
void print_last_error(const char* prefix) {
    const char* error = AI_GetLastError();
    std::cerr << prefix << ": " << error << "\n";
}

// 运行固定次数基准测试，同时记录外层耗时和 DLL 内部耗时。
void run_benchmark(const char* name, int32_t module, int loops, const std::function<std::string()>& fn) {
    constexpr int warmup = 5;
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    double wall_sum = 0.0;
    double wall_min = std::numeric_limits<double>::max();
    double wall_max = 0.0;
    double dll_sum = 0.0;
    double dll_min = std::numeric_limits<double>::max();
    double dll_max = 0.0;
    std::string last;

    for (int i = 0; i < loops; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        last = fn();
        const auto end = std::chrono::high_resolution_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double dll_ms = static_cast<double>(AI_GetLastLatencyUs(module)) / 1000.0;

        wall_sum += wall_ms;
        wall_min = std::min(wall_min, wall_ms);
        wall_max = std::max(wall_max, wall_ms);
        dll_sum += dll_ms;
        dll_min = std::min(dll_min, dll_ms);
        dll_max = std::max(dll_max, dll_ms);
    }

    std::cout << name
              << ": wall_avg=" << (wall_sum / loops) << "ms"
              << " wall_min=" << wall_min << "ms"
              << " wall_max=" << wall_max << "ms"
              << " | dll_avg=" << (dll_sum / loops) << "ms"
              << " dll_min=" << dll_min << "ms"
              << " dll_max=" << dll_max << "ms"
              << " | last=" << last << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int loops = argc > 1 ? std::max(1, std::atoi(argv[1])) : 50;
    std::cout << "version=" << AI_GetVersion() << "\n";
    std::cout << "has_embedded_assets=" << AI_HasEmbeddedAssets() << "\n";

    const auto load_start = std::chrono::high_resolution_clock::now();
    int32_t yolo_handle = 0;
    int32_t load_status = AI_YoloCreate(&yolo_handle);
    if (load_status >= 0) {
        const std::string yolo_model_path = std::filesystem::absolute(
            std::filesystem::u8path(u8"models/yolo/smc.onnx")).u8string();
        const std::string yolo_labels_path = std::filesystem::absolute(
            std::filesystem::u8path(u8"models/yolo/smc.txt")).u8string();
        load_status = YOLO_LoadModelFromPath(
            yolo_handle, yolo_model_path.c_str(), yolo_labels_path.c_str(),
            0, AI_DEVICE_AUTO, 0, 1);
    }
    if (load_status >= 0) load_status = AI_OcrLoadEmbeddedModels(AI_DEVICE_AUTO);
    const auto load_end = std::chrono::high_resolution_clock::now();
    if (load_status < 0) {
        print_last_error("load_embedded_models_failed");
        return 2;
    }
    std::cout << "load_embedded_models_ms="
              << std::chrono::duration<double, std::milli>(load_end - load_start).count()
              << "\n";

    std::vector<uint8_t> ocr_bytes;
    std::vector<uint8_t> yolo_bytes;
    AIImage ocr_image{};
    AIImage yolo_image{};
    if (!image_from_bmp("examples/ocr_test_abc123.bmp", &ocr_bytes, &ocr_image)) {
        std::cerr << "failed_to_read_ocr_image\n";
        return 3;
    }
    if (!image_from_bmp(u8"tests/fixtures/yolo/1.bmp", &yolo_bytes, &yolo_image)) {
        std::cerr << "failed_to_read_yolo_image\n";
        return 4;
    }

    run_benchmark("cv_mean_color_sample_sxr", AI_MODULE_CV, loops, [&]() {
        AIColorStats stats{};
        const int32_t status = AI_CvMeanColor(&yolo_image, nullptr, &stats);
        return status < 0 ? std::string("err") : std::to_string(stats.pixels);
    });
    run_benchmark("cv_find_color_sample_sxr", AI_MODULE_CV, loops, [&]() {
        AIColorFindResult result{};
        const int32_t status = AI_CvFindColor(&yolo_image, nullptr, 0x000000u, 8, &result);
        return status < 0 ? std::string("err") : std::to_string(result.count);
    });
    run_benchmark("cv_extract_trace_ocr_test", AI_MODULE_CV, loops, [&]() {
        std::vector<char> output(262144);
        const int32_t status = AI_CvExtractTraceJson(&ocr_image, nullptr, -1, 1, 1024, output.data(), static_cast<int32_t>(output.size()));
        return status < 0 ? std::string("err") : std::to_string(status);
    });
    run_benchmark("ocr_recognize_line_abc123", AI_MODULE_OCR, loops, [&]() {
        char text[256]{};
        const int32_t status = AI_OcrRecognizeLine(&ocr_image, AI_OCR_OUTPUT_TEXT, 0.0f, text, sizeof(text));
        return status < 0 ? std::string("err") : std::string(text);
    });
    run_benchmark("yolo_infer_sample_sxr", AI_MODULE_YOLO, loops, [&]() {
        const char* json = AI_YoloInfer(yolo_handle, &yolo_image, 0.25f);
        if (json == nullptr || json[0] == '\0') return std::string("err");
        return std::string(json);
    });

    AI_YoloRelease(yolo_handle);
    AI_Release();
    return 0;
}
