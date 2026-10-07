#pragma once
#include "image_view.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace ai {
struct YoloLetterbox { float scale, pad_x, pad_y; };
// Exact sampling/padding of the original two-pass implementation. Generate the
// byte lookup with FP32 division, preserving the original input bits.
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
    static const std::array<float, 256> normalized = [] {
        std::array<float, 256> values{};
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(i) / 255.0f;
        return values;
    }();
    const float padding = normalized[114];
    const int red = bgr && channels >= 3 ? 2 : 0;
    const int green = channels >= 3 ? 1 : 0;
    const int blue = bgr || channels == 1 ? 0 : 2;
    const int right = std::min(width, left + resized_w);
    columns.resize(resized_w);
    for (int x = 0; x < resized_w; ++x)
        columns[x] = std::min(static_cast<int>(x / info.scale), image.width - 1) * channels;
    for (int y = 0; y < height; ++y) {
        const int dy = y - top;
        const uint8_t* row = dy >= 0 && dy < resized_h
            ? image_row_ptr(image, std::min(static_cast<int>(dy / info.scale), image.height - 1)) : nullptr;
        float* r = tensor + static_cast<size_t>(y) * width;
        float* g = r + plane;
        float* b = g + plane;
        if (!row) {
            std::fill_n(r, width, padding);
            std::fill_n(g, width, padding);
            std::fill_n(b, width, padding);
            continue;
        }
        std::fill_n(r, left, padding);
        std::fill_n(g, left, padding);
        std::fill_n(b, left, padding);
        for (int x = left; x < right; ++x) {
            const uint8_t* p = row + columns[x - left];
            r[x] = normalized[p[red]];
            g[x] = normalized[p[green]];
            b[x] = normalized[p[blue]];
        }
        std::fill(r + right, r + width, padding);
        std::fill(g + right, g + width, padding);
        std::fill(b + right, b + width, padding);
    }
    return info;
}
} // namespace ai
