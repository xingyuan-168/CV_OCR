#pragma once
#include "image_view.h"
#include <cmath>
#include <vector>

namespace ai {
struct YoloLetterbox { float scale, pad_x, pad_y; };
// Exact sampling/padding of the original two-pass implementation. Division by
// 255 (not multiplication by a reciprocal) preserves FP32 input bits.
inline YoloLetterbox yolo_preprocess(const AIImage& image, int width, int height,
                                    float* tensor, std::vector<int>& columns) {
    YoloLetterbox info{};
    info.scale = std::min(static_cast<float>(width) / image.width,
                          static_cast<float>(height) / image.height);
    const int resized_w = static_cast<int>(std::round(image.width * info.scale));
    const int resized_h = static_cast<int>(std::round(image.height * info.scale));
    info.pad_x = (width - resized_w) / 2.0f;
    info.pad_y = (height - resized_h) / 2.0f;
    const int left = static_cast<int>(info.pad_x), top = static_cast<int>(info.pad_y);
    const int channels = channels_for_format(image.format);
    const bool bgr = image.format == AI_IMAGE_BGR24 || image.format == AI_IMAGE_BGRA32;
    const size_t plane = static_cast<size_t>(width) * height;
    columns.resize(resized_w);
    for (int x = 0; x < resized_w; ++x)
        columns[x] = std::min(static_cast<int>(x / info.scale), image.width - 1) * channels;
    for (int y = 0; y < height; ++y) {
        const int dy = y - top;
        const uint8_t* row = dy >= 0 && dy < resized_h
            ? image_row_ptr(image, std::min(static_cast<int>(dy / info.scale), image.height - 1)) : nullptr;
        for (int x = 0; x < width; ++x) {
            float r = 114 / 255.0f, g = r, b = r;
            const int dx = x - left;
            if (row && dx >= 0 && dx < resized_w) {
                const uint8_t* p = row + columns[dx];
                r = p[bgr && channels >= 3 ? 2 : 0] / 255.0f;
                g = p[channels >= 3 ? 1 : 0] / 255.0f;
                b = p[bgr || channels == 1 ? 0 : 2] / 255.0f;
            }
            const size_t i = static_cast<size_t>(y) * width + x;
            tensor[i] = r; tensor[plane + i] = g; tensor[2 * plane + i] = b;
        }
    }
    return info;
}
} // namespace ai
