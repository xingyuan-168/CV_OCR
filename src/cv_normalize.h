#pragma once
namespace ai::cv_detail {
int prepare_row_avx2(const unsigned char *input, int count, float *const *planes, int *const *sums,
                     const int *const *previous, double *square, const double *previous_square, double *carry,
                     bool centered);
void spectrum_sum_avx2(const float *const *images, const float *const *templates, float *out, int rows,
                       int cols);
int normalize_avx2(const int *const *top, const int *const *bottom, const double *qtop, const double *qbottom,
                   const float *raw, float *out, int count, int width, const double *sums, double inv_area,
                   double norm, bool centered, float floor);
} // namespace ai::cv_detail
