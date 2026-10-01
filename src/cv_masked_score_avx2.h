#pragma once

#include <cstddef>
#include <cstdint>

namespace ai::cv_detail {

struct MaskedAccumulators {
    uint64_t sums[3]{};
    uint64_t squares[3]{};
    uint64_t dots[3]{};
    bool identical = true;
};

struct TilePeak {
    float score;
    uint32_t index;
};

TilePeak tile_peak_avx2(const float* scores, size_t score_stride,
    const uint8_t* suppressed, int columns,
    int start_x, int start_y, int end_x, int end_y);

// The caller guarantees count <= 65535 and ascending signed-32-bit offsets.
// The final visible pixel is processed scalarly, keeping four-byte gathers
// within the image even when the last visible pixel touches its final byte.
MaskedAccumulators masked_accumulate_avx2(
    const uint8_t* image_origin, const uint32_t* offsets,
    const uint32_t* template_bgr, size_t count);

} // namespace ai::cv_detail
