#pragma once

#include "ai_engine.h"

#include <algorithm>
#include <cstddef>
#include <stdint.h>

namespace ai {

// 返回 AI_IMAGE_* 格式每个像素的字节数；未知格式返回 0。
inline int channels_for_format(int32_t format) {
    switch (format) {
        case AI_IMAGE_GRAY8:
            return 1;
        case AI_IMAGE_BGR24:
        case AI_IMAGE_RGB24:
            return 3;
        case AI_IMAGE_BGRA32:
        case AI_IMAGE_RGBA32:
            return 4;
        default:
            return 0;
    }
}

// 校验读取 AIImage 前必须满足的最小条件。
// 允许负 stride，但其绝对值必须覆盖完整一行。
inline bool validate_image(const AIImage& image) {
    const int channels = channels_for_format(image.format);
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 || channels == 0) {
        return false;
    }
    const int64_t stride = static_cast<int64_t>(image.stride);
    const int64_t abs_stride = stride < 0 ? -stride : stride;
    return abs_stride >= static_cast<int64_t>(image.width) * channels;
}

// 返回逻辑第 y 行的只读指针，同时支持正/负 stride。
inline const uint8_t* image_row_ptr(const AIImage& image, int32_t y) {
    return image.data + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(image.stride);
}

// 可写重载，用于需要写入 AIImage 缓冲区的操作。
inline uint8_t* image_row_ptr(AIImage& image, int32_t y) {
    return image.data + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(image.stride);
}

// 将可选 ROI 裁剪到图像范围内；空 ROI 表示整张图。
inline AIRect normalize_roi(const AIImage& image, const AIRect* roi) {
    AIRect out{0, 0, image.width, image.height};
    if (roi != nullptr) {
        out = *roi;
    }

    const int32_t x1 = std::max<int32_t>(0, out.x);
    const int32_t y1 = std::max<int32_t>(0, out.y);
    const int32_t x2 = std::min<int32_t>(image.width, out.x + std::max<int32_t>(0, out.w));
    const int32_t y2 = std::min<int32_t>(image.height, out.y + std::max<int32_t>(0, out.h));
    return AIRect{x1, y1, std::max<int32_t>(0, x2 - x1), std::max<int32_t>(0, y2 - y1)};
}

// 将任意支持的像素格式读取为 B/G/R 通道顺序。
inline void read_bgr(const uint8_t* pixel, int32_t format, uint8_t* b, uint8_t* g, uint8_t* r) {
    switch (format) {
        case AI_IMAGE_GRAY8:
            *b = *g = *r = pixel[0];
            break;
        case AI_IMAGE_BGR24:
        case AI_IMAGE_BGRA32:
            *b = pixel[0];
            *g = pixel[1];
            *r = pixel[2];
            break;
        case AI_IMAGE_RGB24:
        case AI_IMAGE_RGBA32:
            *r = pixel[0];
            *g = pixel[1];
            *b = pixel[2];
            break;
        default:
            *b = *g = *r = 0;
            break;
    }
}

// 四通道格式返回 alpha；无 alpha 的格式视为完全不透明。
inline uint8_t read_alpha(const uint8_t* pixel, int32_t format) {
    switch (format) {
        case AI_IMAGE_BGRA32:
        case AI_IMAGE_RGBA32:
            return pixel[3];
        default:
            return 255;
    }
}

// 快速整数灰度转换，权重接近 ITU-R BT.601 亮度公式。
inline uint8_t bgr_to_gray(uint8_t b, uint8_t g, uint8_t r) {
    return static_cast<uint8_t>((29 * b + 150 * g + 77 * r) >> 8);
}

} // namespace ai
