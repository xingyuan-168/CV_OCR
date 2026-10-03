#pragma once
#include "image_view.h"
#include <cstring>
#include <limits>

namespace ai {
// Read-only zero-copy 24-bit BI_RGB view. data points to logical top row;
// negative stride represents bottom-up BMP without copying or flipping it.
inline bool bmp24_view(const uint8_t* data, size_t size, AIImage* image) {
    if (!data || !image || size < 54 || data[0] != 'B' || data[1] != 'M') return false;
    auto u32 = [&](size_t p) { uint32_t v; std::memcpy(&v, data + p, 4); return v; };
    auto u16 = [&](size_t p) { uint16_t v; std::memcpy(&v, data + p, 2); return v; };
    const uint64_t offset = u32(10), dib = u32(14);
    const int32_t width = static_cast<int32_t>(u32(18)), raw_height = static_cast<int32_t>(u32(22));
    if (dib < 40 || 14 + dib > offset || offset >= size || width <= 0 || raw_height == 0 ||
        raw_height == std::numeric_limits<int32_t>::min() || u16(26) != 1 || u16(28) != 24 || u32(30) != 0) return false;
    const int64_t height = raw_height < 0 ? -static_cast<int64_t>(raw_height) : raw_height;
    const int64_t stride = (static_cast<int64_t>(width) * 3 + 3) & ~int64_t(3);
    if (stride > INT32_MAX || height > static_cast<int64_t>((size - offset) / stride)) return false;
    const uint8_t* pixels = data + offset;
    if (raw_height > 0) pixels += static_cast<size_t>(height - 1) * stride;
    *image = {const_cast<uint8_t*>(pixels), width, static_cast<int32_t>(height), static_cast<int32_t>(raw_height > 0 ? -stride : stride), AI_IMAGE_BGR24};
    return true;
}
} // namespace ai
