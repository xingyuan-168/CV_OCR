#include "cv_masked_score_avx2.h"

#include <immintrin.h>
#include <intrin.h>

namespace ai::cv_detail {

TilePeak tile_peak_avx2(const float* scores, size_t score_stride,
    const uint8_t* suppressed, int columns,
    int start_x, int start_y, int end_x, int end_y) {
    TilePeak best{-1.0f, static_cast<uint32_t>(
        static_cast<size_t>(start_y) * columns + start_x)};
    const __m128i zero = _mm_setzero_si128();
    for (int y = start_y; y < end_y; ++y) {
        const float* row = scores + static_cast<size_t>(y) * score_stride;
        const uint8_t* mask = suppressed + static_cast<size_t>(y) * columns;
        int x = start_x;
        for (; x + 8 <= end_x; x += 8) {
            const __m128i flags = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(mask + x));
            const unsigned available = static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpeq_epi8(flags, zero))) & 255u;
            if (!available) continue;
            const __m256 values = _mm256_loadu_ps(row + x);
            const __m256 higher_mask = _mm256_cmp_ps(
                values, _mm256_set1_ps(best.score), _CMP_GT_OQ);
            const unsigned higher = static_cast<unsigned>(_mm256_movemask_ps(higher_mask));
            if (!higher) continue;
            unsigned eligible = higher & available;
            if (eligible) {
                float maximum;
                if (eligible & (eligible - 1)) {
                    const __m256 permitted = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                        _mm256_cvtepu8_epi32(flags), _mm256_setzero_si256()));
                    const __m256 valid = _mm256_blendv_ps(_mm256_set1_ps(-1.0f),
                        values, _mm256_and_ps(permitted, higher_mask));
                    __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(valid),
                        _mm256_extractf128_ps(valid, 1));
                    reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
                    reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
                    maximum = _mm_cvtss_f32(reduced);
                    eligible &= static_cast<unsigned>(_mm256_movemask_ps(
                        _mm256_cmp_ps(values, _mm256_set1_ps(maximum), _CMP_EQ_OQ)));
                } else {
                    unsigned long single = 0;
                    _BitScanForward(&single, eligible);
                    maximum = row[x + single];
                }
                unsigned long lane = 0;
                _BitScanForward(&lane, eligible);
                const int column = x + static_cast<int>(lane);
                best = {maximum, static_cast<uint32_t>(
                    static_cast<size_t>(y) * columns + column)};
            }
        }
        for (; x < end_x; ++x) {
            if (!mask[x] && row[x] > best.score) {
                best = {row[x], static_cast<uint32_t>(
                    static_cast<size_t>(y) * columns + x)};
            }
        }
    }
    return best;
}

MaskedAccumulators masked_accumulate_avx2(
    const uint8_t* image_origin, const uint32_t* offsets,
    const uint32_t* template_bgr, size_t count) {
    __m256i sums[3] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
    __m256i squares[3] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
    __m256i dots[3] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
    __m256i difference = _mm256_setzero_si256();
    const __m256i byte_mask = _mm256_set1_epi32(255);
    const __m256i color_mask = _mm256_set1_epi32(0x00ffffff);
    size_t i = 0;
    for (; i + 8 < count; i += 8) {
        const __m256i positions = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(offsets + i));
        const __m256i observed = _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(image_origin), positions, 1);
        const __m256i expected = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(template_bgr + i));
        difference = _mm256_or_si256(difference,
            _mm256_and_si256(_mm256_xor_si256(observed, expected), color_mask));
        const __m256i values[3] = {
            _mm256_and_si256(observed, byte_mask),
            _mm256_and_si256(_mm256_srli_epi32(observed, 8), byte_mask),
            _mm256_and_si256(_mm256_srli_epi32(observed, 16), byte_mask)};
        const __m256i templates[3] = {
            _mm256_and_si256(expected, byte_mask),
            _mm256_and_si256(_mm256_srli_epi32(expected, 8), byte_mask),
            _mm256_and_si256(_mm256_srli_epi32(expected, 16), byte_mask)};
        for (int c = 0; c < 3; ++c) {
            const __m256i value = values[c];
            const __m256i templ = templates[c];
            sums[c] = _mm256_add_epi32(sums[c], value);
            squares[c] = _mm256_add_epi32(squares[c], _mm256_mullo_epi32(value, value));
            dots[c] = _mm256_add_epi32(dots[c], _mm256_mullo_epi32(value, templ));
        }
    }
    MaskedAccumulators result;
    alignas(32) uint32_t lanes[8];
    for (int c = 0; c < 3; ++c) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sums[c]);
        for (uint32_t value : lanes) result.sums[c] += value;
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), squares[c]);
        for (uint32_t value : lanes) result.squares[c] += value;
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), dots[c]);
        for (uint32_t value : lanes) result.dots[c] += value;
    }
    result.identical = _mm256_testz_si256(difference, difference) != 0;
    for (; i < count; ++i) {
        const uint8_t* pixel = image_origin + offsets[i];
        const uint32_t templ = template_bgr[i];
        for (int c = 0; c < 3; ++c) {
            const uint32_t value = pixel[c];
            const uint32_t expected = (templ >> (c * 8)) & 255u;
            result.identical &= value == expected;
            result.sums[c] += value;
            result.squares[c] += value * value;
            result.dots[c] += value * expected;
        }
    }
    return result;
}

} // namespace ai::cv_detail
