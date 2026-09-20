#include "cv_normalize.h"
#include <cfloat>
#include <immintrin.h>

namespace ai::cv_detail {
int prepare_row_avx2(const unsigned char *input, int count, float *const *planes, int *const *sums,
                     const int *const *previous, double *square, const double *previous_square, double *carry,
                     bool centered) {
    const auto indices = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const auto mask = _mm256_set1_epi32(255);
    auto prefix = [](__m256i value) {
        value = _mm256_add_epi32(value, _mm256_slli_si256(value, 4));
        value = _mm256_add_epi32(value, _mm256_slli_si256(value, 8));
        const auto lower = _mm256_castsi256_si128(value);
        return _mm256_add_epi32(
            value, _mm256_inserti128_si256(_mm256_setzero_si256(), _mm_shuffle_epi32(lower, 255), 1));
    };
    int x = 0;
    // Gather reads four bytes at each BGR pixel. Keep the last block scalar so
    // the final three-byte pixel is never read beyond the input row.
    for (; x + 8 < count; x += 8) {
        const auto packed = _mm256_i32gather_epi32(reinterpret_cast<const int *>(input + x * 3), indices, 1);
        const __m256i values[3] = {_mm256_and_si256(packed, mask),
                                   _mm256_and_si256(_mm256_srli_epi32(packed, 8), mask),
                                   _mm256_and_si256(_mm256_srli_epi32(packed, 16), mask)};
        auto q = _mm256_setzero_si256();
        for (int c = 0; c < 3; ++c) {
            _mm256_storeu_ps(planes[c] + x, _mm256_cvtepi32_ps(values[c]));
            q = _mm256_add_epi32(q, _mm256_mullo_epi32(values[c], values[c]));
            if (centered) {
                const auto sum =
                    _mm256_add_epi32(prefix(values[c]), _mm256_set1_epi32(static_cast<int>(carry[c])));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(sums[c] + x + 1),
                    _mm256_add_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(previous[c] + x + 1)), sum));
                carry[c] = _mm256_extract_epi32(sum, 7);
            }
        }
        q = prefix(q);
        const auto running = _mm256_set1_pd(carry[3]);
        _mm256_storeu_pd(
            square + x + 1,
            _mm256_add_pd(_mm256_loadu_pd(previous_square + x + 1),
                          _mm256_add_pd(running, _mm256_cvtepi32_pd(_mm256_castsi256_si128(q)))));
        _mm256_storeu_pd(
            square + x + 5,
            _mm256_add_pd(_mm256_loadu_pd(previous_square + x + 5),
                          _mm256_add_pd(running, _mm256_cvtepi32_pd(_mm256_extracti128_si256(q, 1)))));
        carry[3] += _mm256_extract_epi32(q, 7);
    }
    return x;
}
void spectrum_sum_avx2(const float *const *images, const float *const *templates, float *out, int rows,
                       int cols) {
    const auto sign = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    const int end = (cols % 2 == 0) ? cols - 1 : cols;
    for (int y = 0; y < rows; ++y) {
        int x = 1;
        for (; x + 4 <= end; x += 4) {
            __m128 total = _mm_setzero_ps();
            for (int c = 0; c < 3; ++c) {
                const auto a = _mm256_cvtps_pd(_mm_loadu_ps(images[c] + y * cols + x));
                const auto b = _mm256_cvtps_pd(_mm_loadu_ps(templates[c] + y * cols + x));
                const auto first = _mm256_mul_pd(a, _mm256_movedup_pd(b));
                const auto second = _mm256_mul_pd(_mm256_permute_pd(a, 5), _mm256_permute_pd(b, 15));
                total = _mm_add_ps(total, _mm256_cvtpd_ps(_mm256_add_pd(first, _mm256_xor_pd(second, sign))));
            }
            _mm_storeu_ps(out + y * cols + x, total);
        }
        for (; x < end; x += 2) {
            float real = 0, imag = 0;
            for (int c = 0; c < 3; ++c) {
                const float *a = images[c] + y * cols + x;
                const float *b = templates[c] + y * cols + x;
                real +=
                    static_cast<float>(static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1]);
                imag +=
                    static_cast<float>(static_cast<double>(a[1]) * b[0] - static_cast<double>(a[0]) * b[1]);
            }
            out[y * cols + x] = real;
            out[y * cols + x + 1] = imag;
        }
    }
    for (int x = 0; x < cols; x += (cols % 2 == 0 ? cols - 1 : cols)) {
        out[x] = 0;
        for (int c = 0; c < 3; ++c)
            out[x] += images[c][x] * templates[c][x];
        int y = 1;
        for (; y + 1 < rows; y += 2) {
            float real = 0, imag = 0;
            for (int c = 0; c < 3; ++c) {
                const double ar = images[c][y * cols + x], ai = images[c][(y + 1) * cols + x];
                const double br = templates[c][y * cols + x], bi = templates[c][(y + 1) * cols + x];
                real += static_cast<float>(ar * br + ai * bi);
                imag += static_cast<float>(ai * br - ar * bi);
            }
            out[y * cols + x] = real;
            out[(y + 1) * cols + x] = imag;
        }
        if (y < rows) {
            out[y * cols + x] = 0;
            for (int c = 0; c < 3; ++c)
                out[y * cols + x] += images[c][y * cols + x] * templates[c][y * cols + x];
        }
    }
}
int normalize_avx2(const int *const *top, const int *const *bottom, const double *qtop, const double *qbottom,
                   const float *raw, float *out, int count, int width, const double *sums, double inv_area,
                   double norm, bool centered, float floor) {
    const auto zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1), neg = _mm256_set1_pd(-1);
    const auto inv = _mm256_set1_pd(inv_area), scale = _mm256_set1_pd(norm);
    const auto sign = _mm256_set1_pd(-0.0);
    const auto window = [width](const double *a, const double *b, int x) {
        return _mm256_add_pd(
            _mm256_sub_pd(_mm256_sub_pd(_mm256_loadu_pd(a + x), _mm256_loadu_pd(a + x + width)),
                          _mm256_loadu_pd(b + x)),
            _mm256_loadu_pd(b + x + width));
    };
    int x = 0;
    for (; x + 4 <= count; x += 4) {
        auto num = _mm256_cvtps_pd(_mm_loadu_ps(raw + x));
        auto mean2 = zero;
        if (centered)
            for (int c = 0; c < 3; ++c) {
                const auto *a = top[c] + x;
                const auto *b = bottom[c] + x;
                const auto load = [](const int *p) {
                    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
                };
                const auto v = _mm256_cvtepi32_pd(_mm_add_epi32(
                    _mm_sub_epi32(_mm_sub_epi32(load(a), load(a + width)), load(b)), load(b + width)));
                num = _mm256_sub_pd(num, _mm256_mul_pd(_mm256_mul_pd(v, _mm256_set1_pd(sums[c])), inv));
                mean2 = _mm256_add_pd(mean2, _mm256_mul_pd(v, v));
            }
        const auto square = window(qtop, qbottom, x);
        const auto variance = _mm256_max_pd(zero, _mm256_sub_pd(square, _mm256_mul_pd(mean2, inv)));
        const auto cutoff =
            _mm256_min_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(_mm256_set1_pd(10 * FLT_EPSILON), square));
        if (floor > 0) {
            const auto bound =
                _mm256_mul_pd(variance, _mm256_set1_pd(static_cast<double>(floor) * floor * norm * norm));
            const auto possible =
                _mm256_and_pd(_mm256_cmp_pd(variance, cutoff, _CMP_GT_OQ),
                              _mm256_and_pd(_mm256_cmp_pd(num, zero, _CMP_GT_OQ),
                                            _mm256_cmp_pd(_mm256_mul_pd(num, num), bound, _CMP_GE_OQ)));
            if (_mm256_movemask_pd(possible) == 0) {
                _mm_storeu_ps(out + x, _mm_set1_ps(-1));
                continue;
            }
        }
        auto denominator = _mm256_mul_pd(_mm256_sqrt_pd(variance), scale);
        denominator = _mm256_and_pd(denominator, _mm256_cmp_pd(variance, cutoff, _CMP_GT_OQ));
        const auto absolute = _mm256_andnot_pd(sign, num);
        const auto normal = _mm256_cmp_pd(absolute, denominator, _CMP_LT_OQ);
        const auto saturated =
            _mm256_cmp_pd(absolute, _mm256_mul_pd(denominator, _mm256_set1_pd(1.125)), _CMP_LT_OQ);
        const auto unit = _mm256_blendv_pd(neg, one, _mm256_cmp_pd(num, zero, _CMP_GT_OQ));
        const auto safe_denominator = _mm256_blendv_pd(one, denominator, normal);
        const auto value =
            _mm256_blendv_pd(_mm256_and_pd(unit, saturated), _mm256_div_pd(num, safe_denominator), normal);
        _mm_storeu_ps(out + x, _mm256_cvtpd_ps(value));
    }
    return x;
}
} // namespace ai::cv_detail
