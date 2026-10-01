#include "cv_masked_score_avx2.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {
uint32_t next_random(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

void check(size_t count, bool identical, bool all_maximum) {
    std::vector<uint8_t> image(count * 4 + 8);
    std::vector<uint32_t> offsets(count), templ(count);
    uint32_t state = 0x51a3f987u;
    for (size_t i = 0; i < count; ++i) {
        offsets[i] = static_cast<uint32_t>(i * 4);
        for (int c = 0; c < 3; ++c) {
            const uint8_t value = all_maximum ? 255u :
                static_cast<uint8_t>(next_random(state) >> 24);
            image[i * 4 + c] = value;
            const uint8_t expected = identical ? value :
                static_cast<uint8_t>(next_random(state) >> 24);
            templ[i] |= static_cast<uint32_t>(expected) << (c * 8);
        }
        image[i * 4 + 3] = static_cast<uint8_t>(next_random(state) >> 24);
    }
    const auto actual = ai::cv_detail::masked_accumulate_avx2(
        image.data(), offsets.data(), templ.data(), count);
    ai::cv_detail::MaskedAccumulators expected;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* pixel = image.data() + offsets[i];
        for (int c = 0; c < 3; ++c) {
            const uint32_t value = pixel[c];
            const uint32_t target = (templ[i] >> (c * 8)) & 255u;
            expected.identical &= value == target;
            expected.sums[c] += value;
            expected.squares[c] += value * value;
            expected.dots[c] += value * target;
        }
    }
    assert(actual.identical == expected.identical);
    for (int c = 0; c < 3; ++c) {
        assert(actual.sums[c] == expected.sums[c]);
        assert(actual.squares[c] == expected.squares[c]);
        assert(actual.dots[c] == expected.dots[c]);
    }
}
void check_peaks() {
    uint32_t state = 0xf38a6791u;
    for (int columns : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
        for (int rows : {1, 7, 8, 9, 16, 17}) {
            // A padded score stride also covers cv::Mat views.
            const size_t stride = static_cast<size_t>(columns) + 3;
            std::vector<float> scores(stride * rows);
            std::vector<uint8_t> suppressed(static_cast<size_t>(columns) * rows);
            for (int sample = 0; sample < 32; ++sample) {
                for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) {
                    const uint32_t random = next_random(state);
                    float value = static_cast<float>(random % 13) / 13.0f;
                    if (sample == 0) value = 0.5f; // exact ties
                    if (sample == 1 || (random & 31u) == 0)
                        value = std::numeric_limits<float>::quiet_NaN();
                    if (sample == 2) value = -1.0f;
                    scores[static_cast<size_t>(y) * stride + x] = value;
                    suppressed[static_cast<size_t>(y) * columns + x] =
                        sample == 3 || ((random >> 8) & 3u) == 0 ? 1 : 0;
                }
                for (int sy = 0; sy < rows; sy += 8) for (int sx = 0; sx < columns; sx += 16) {
                    const int ey = std::min(sy + 8, rows);
                    const int ex = std::min(sx + 16, columns);
                    ai::cv_detail::TilePeak expected{-1.0f,
                        static_cast<uint32_t>(sy * columns + sx)};
                    for (int y = sy; y < ey; ++y) for (int x = sx; x < ex; ++x) {
                        const float value = scores[static_cast<size_t>(y) * stride + x];
                        if (!suppressed[static_cast<size_t>(y) * columns + x] &&
                            value > expected.score) expected = {value, static_cast<uint32_t>(y * columns + x)};
                    }
                    const auto actual = ai::cv_detail::tile_peak_avx2(
                        scores.data(), stride, suppressed.data(), columns, sx, sy, ex, ey);
                    assert(actual.score == expected.score && actual.index == expected.index);
                }
            }
        }
    }
}
} // namespace

int main() {
    if (!cv::checkHardwareSupport(CV_CPU_AVX2)) {
        std::cout << "AVX2 unavailable; scalar integration is covered by CV tests\n";
        return 0;
    }
    for (size_t count : std::vector<size_t>{1, 2, 7, 8, 9, 16, 17, 367, 22017, 65535}) {
        check(count, true, false);
        check(count, false, false);
        check(count, true, true);
        check(count, false, true);
    }
    check_peaks();
    std::cout << "masked AVX2 integer accumulators PASS\n";
}
