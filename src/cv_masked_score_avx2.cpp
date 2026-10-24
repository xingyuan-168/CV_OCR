#include "cv_masked_score_avx2.h"

#include <immintrin.h>
#include <intrin.h>
#include <cfloat>

namespace ai::cv_detail {

int masked_projection_candidates_avx2(const uint8_t* origin,
        const uint32_t* offsets,const int32_t* differences,const MaskedProjectionLimit* stages,
        size_t stage_count,int columns,float* output) {
    const auto zero=_mm256_setzero_si256();const auto zero_f=_mm256_setzero_ps();
    int x=0;
    // Eight adjacent planar bytes correspond to eight legal windows, including
    // the final group at the right edge. No four-byte pixel gather is used.
    for(;x+8<=columns;x+=8) {
        auto squares=zero,dots=zero;auto excluded=zero_f;size_t k=0;
        for(size_t level=0;level<stage_count&&_mm256_movemask_ps(excluded)!=255;++level) {
            const auto& stage=stages[level];
            for(;k<stage.end;++k){const auto*a=origin+x+offsets[2*k],*b=origin+x+offsets[2*k+1];
                const auto va=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a)));
                const auto vb=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b)));
                const auto v=_mm256_sub_epi32(vb,va);
                squares=_mm256_add_epi32(squares,_mm256_mullo_epi32(v,v));
                dots=_mm256_add_epi32(dots,_mm256_mullo_epi32(v,_mm256_set1_epi32(differences[k])));
            }
            // Q and C are exact signed32 integer accumulations: at most1536
            // disjoint channel pairs, each contribution at most65025. There
            // is no subtractive covariance cancellation. Enclose conversion
            // and multiplication with outward factors before rejecting.
            const auto q=_mm256_cvtepi32_ps(squares),c=_mm256_max_ps(zero_f,_mm256_cvtepi32_ps(dots));
            const auto lhs=_mm256_mul_ps(c,c);
            const auto rhs=_mm256_mul_ps(q,_mm256_set1_ps(stage.vector));
            excluded=_mm256_or_ps(excluded,_mm256_cmp_ps(lhs,rhs,_CMP_LT_OQ));
            excluded=_mm256_or_ps(excluded,_mm256_castsi256_ps(_mm256_cmpeq_epi32(squares,zero)));
        }
        _mm256_storeu_ps(output+x,_mm256_blendv_ps(_mm256_set1_ps(1),_mm256_set1_ps(-1),excluded));
    }return x;
}

int masked_pair_candidates_avx2(const uint8_t* origin, int channels, unsigned selected_channels,
        const uint32_t* offsets, size_t pairs, const float* numerator, double error,
        double reject_factor, int columns, float* output) {
    const auto indices = _mm256_setr_epi32(0, channels, channels * 2, channels * 3,
        channels * 4, channels * 5, channels * 6, channels * 7);
    const auto byte_mask = _mm256_set1_epi32(255);
    const auto delta = _mm256_set1_ps(static_cast<float>(error * (1.0 + 1e-6) + 1e-6));
    const auto factor = _mm256_set1_ps(static_cast<float>(reject_factor * (1.0 - 1e-6)));
    const auto minus_one = _mm256_set1_ps(-1), one = _mm256_set1_ps(1);
    int x = 0;
    // Leave the last window to the scalar tail: BGR gathers read four bytes.
    for (; x + 8 < columns; x += 8) {
        const auto upper = _mm256_add_ps(_mm256_loadu_ps(numerator + x), delta);
        const auto square = _mm256_mul_ps(_mm256_mul_ps(upper, upper), _mm256_set1_ps(1.000001f));
        auto excluded = _mm256_cmp_ps(upper, _mm256_setzero_ps(), _CMP_LT_OQ);
        auto square_sum = _mm256_setzero_si256();
        __m256i sums[3] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (size_t k = 0; k < pairs && _mm256_movemask_ps(excluded) != 255; ++k) {
            const auto* first = origin + x * channels + offsets[k * 2];
            const auto* second = origin + x * channels + offsets[k * 2 + 1];
            __m256i va, vb;
            if (channels == 1) {
                va = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(first)));
                vb = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(second)));
            } else {
                va = _mm256_i32gather_epi32(reinterpret_cast<const int*>(first), indices, 1);
                vb = _mm256_i32gather_epi32(reinterpret_cast<const int*>(second), indices, 1);
            }
            for (int c = 0; c < channels; ++c) {
                if (selected_channels & (1u << c)) {
                    const auto a = _mm256_and_si256(va, byte_mask), b = _mm256_and_si256(vb, byte_mask);
                    sums[c] = _mm256_add_epi32(sums[c], _mm256_add_epi32(a, b));
                    square_sum = _mm256_add_epi32(square_sum,
                        _mm256_add_epi32(_mm256_mullo_epi32(a, a), _mm256_mullo_epi32(b, b)));
                }
                va = _mm256_srli_epi32(va, 8);
                vb = _mm256_srli_epi32(vb, 8);
            }
            const auto q = _mm256_cvtepi32_ps(square_sum);
            auto lower = q;
            const auto inverse = _mm256_set1_ps(1.0f / float((k + 1) * 2));
            for (int c = 0; c < channels; ++c) if (selected_channels & (1u << c)) {
                const auto sum = _mm256_cvtepi32_ps(sums[c]);
                lower = _mm256_sub_ps(lower, _mm256_mul_ps(_mm256_mul_ps(sum, sum), inverse));
            }
            // Round the lower bound down, so cancellation cannot discard a hit.
            lower = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(lower,
                _mm256_add_ps(_mm256_mul_ps(q, _mm256_set1_ps(2e-6f)), _mm256_set1_ps(1))));
            excluded = _mm256_or_ps(excluded,
                _mm256_cmp_ps(square, _mm256_mul_ps(factor, lower), _CMP_LT_OQ));
        }
        _mm256_storeu_ps(output + x, _mm256_blendv_ps(one, minus_one, excluded));
    }
    return x;
}

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
