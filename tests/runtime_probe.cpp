#include "ai_engine.h"

#include <cstdint>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// 将完整二进制文件读入内存，用于模型和 BMP 加载测试。
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

// 将未压缩 24/32 位 BMP 解码为 AIImage 视图，并由 keepalive 字节保持生命周期。
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

} // namespace

// 针对内存加载 OCR/YOLO 模型的端到端运行探测。
int main(int argc, char** argv) {
    std::cout << AI_GetVersion() << "\n";

    std::vector<uint8_t> ocr_image_bytes;
    AIImage ocr_image{};
    if (!image_from_bmp("examples/ocr_test_abc123.bmp", &ocr_image_bytes, &ocr_image)) {
        std::cerr << "无法读取 OCR 示例图片\n";
        return 2;
    }

    const bool use_embedded_assets = AI_HasEmbeddedAssets() != 0;
    int status = AI_ERR_RUNTIME;
    // OCR 资源可单独内置；先尝试内置加载，再回退到显式外部模型。
    status = AI_OcrLoadEmbeddedModels(AI_DEVICE_AUTO);
    if (status < 0 && !use_embedded_assets) {
        const char* ocr_model_path = argc > 1 ? argv[1] : "models/ocr_ppocrv6/rec.fp32.onnx";
        std::vector<uint8_t> ocr_model = read_file(ocr_model_path);
        if (ocr_model.empty()) {
            std::cerr << "无法读取 OCR 模型\n";
            return 2;
        }
        status = AI_OcrLoadModelsFromMemory(nullptr, 0, ocr_model.data(), static_cast<int32_t>(ocr_model.size()), "configs/ocr_ppocrv6_tiny.ini", AI_DEVICE_AUTO);
    }
    if (status < 0) {
        const char* error = AI_GetLastError();
        std::cerr << "OCR 加载失败：" << status << " " << error << "\n";
        return 3;
    }

    char text[256]{};
    status = AI_OcrRecognizeLine(&ocr_image, AI_OCR_OUTPUT_TEXT, 0.0f, text, sizeof(text));
    if (status < 0) {
        const char* error = AI_GetLastError();
        std::cerr << "OCR 推理失败：" << status << " " << error << "\n";
        return 4;
    }
    std::cout << "ocr_text=" << text << "\n";
    AI_OcrRelease();

    int32_t yolo_handle = 0;
    if (AI_YoloCreate(&yolo_handle) < 0) return 5;
    const std::string yolo_model_path = std::filesystem::absolute(
        std::filesystem::u8path(u8"models/yolo/smc.onnx")).u8string();
    const std::string yolo_labels_path = std::filesystem::absolute(
        std::filesystem::u8path(u8"models/yolo/smc.txt")).u8string();
    status = YOLO_LoadModelFromPath(
        yolo_handle, yolo_model_path.c_str(), yolo_labels_path.c_str(),
        0, AI_DEVICE_AUTO, 0, 1);
    if (status < 0) {
        const char* error = AI_GetLastError();
        std::cerr << "YOLO 加载失败：" << status << " " << error << "\n";
        return 5;
    }

    std::vector<uint8_t> yolo_image_bytes;
    AIImage yolo_image{};
    std::vector<uint8_t> yolo_pixels;
    if (!image_from_bmp("examples/sample_sxr.bmp", &yolo_image_bytes, &yolo_image)) {
        yolo_pixels.assign(static_cast<size_t>(357 * 161 * 3), 0);
        yolo_image = AIImage{yolo_pixels.data(), 357, 161, 357 * 3, AI_IMAGE_BGR24};
    }
    const char* yolo_json = AI_YoloInfer(yolo_handle, &yolo_image, 0.25f);
    if (yolo_json == nullptr || yolo_json[0] == '\0') {
        const char* error = AI_GetLastError();
        std::cerr << "YOLO 推理失败：" << error << "\n";
        return 6;
    }
    std::cout << "yolo_json=" << yolo_json << "\n";
    AI_YoloRelease(yolo_handle);

    return 0;
}
