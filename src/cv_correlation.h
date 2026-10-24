#pragma once

#include <array>
#include <vector>
#include <opencv2/core.hpp>
#include "cv_masked_score_avx2.h"

namespace ai::cv_detail {

// A single geometry per template, owned by the template snapshot, never by a
// workspace. Spectra contain pixels, not results from a previous request.
struct TemplateSpectrum {
    cv::Size geometry;
    std::array<cv::Mat, 3> channels;
};

struct CorrelationWorkspace {
    std::array<cv::Mat, 3> planes;
    std::array<cv::Mat, 3> spectra;
    std::array<cv::Mat, 3> sums;
    cv::Mat square_sum, product, accumulated, correlation;
    cv::Mat transform_plan, transform_buffer;
    cv::Size plan_geometry;
    void transform(const cv::Mat &input, cv::Mat &output, bool inverse);
    std::array<double, 4> timings{};
    void discard() noexcept;
    size_t retained_bytes() const noexcept;
    void prepare(const cv::Mat &image, bool centered = true);
    void match(const cv::Mat &templ, const TemplateSpectrum &spectrum, const double *sums, double energy,
               bool centered, cv::Mat &score, float floor = -1.0f);
    float exact_at(const cv::Mat &image, const cv::Mat &templ, const double *template_sums,
                   double template_energy, int x, int y) const;
};

TemplateSpectrum make_spectrum(const cv::Mat &templ, cv::Size geometry);
cv::Size spectrum_geometry(cv::Size image);

// The numerator uses only visible, per-channel centered template pixels.
// Variance of a visible subset is a lower bound on full masked variance:
// sum_subset((I-mean_subset)^2) <= sum_mask((I-mean_mask)^2). No background pixel
// participates. The resulting upper bound excludes candidates; the
// public score is always recomputed with integer masked accumulators.
struct MaskedTemplateSpectrum {
    cv::Size geometry;
    cv::Size extent;
    int channels = 0;
    unsigned active_channels = 0;
    size_t count = 0;
    double energy = 0, energy_error = 0, numerator_error = 0;
    std::array<cv::Mat, 3> centered;
    std::vector<cv::Point> visible;
    std::vector<std::pair<cv::Point, cv::Point>> pairs;
    std::vector<std::pair<cv::Point, cv::Point>> projection_pairs;
    std::vector<int> projection_channels;
    std::vector<int32_t> projection_differences;
    std::vector<MaskedProjectionStage> projection_stages;
};

struct MaskedCorrelationWorkspace {
    CorrelationWorkspace fft;
    std::array<cv::Mat, 3> spectra;
    cv::Mat plane, product, accumulated, numerator;
    unsigned prepared_channels = 0;
    unsigned constant_channels = 0;
    cv::Size geometry;
    int channels = 0;
    std::vector<uint32_t> projection_offsets;
    cv::Mat projection_planes;
    bool projection_prepared = false;
    bool projection_constant = false;
    void invalidate() noexcept { prepared_channels = constant_channels = 0; projection_prepared=false; }
    void prepare(const cv::Mat& image, const MaskedTemplateSpectrum& templ);
    void upper_bounds(const cv::Mat& image, const MaskedTemplateSpectrum& templ,
                      float threshold, cv::Mat& output);
    void discard() noexcept;
    size_t retained_bytes() const noexcept;
};

MaskedTemplateSpectrum make_masked_spectrum(const cv::Mat& templ, const cv::Mat& mask,
                                           cv::Size geometry);

} // namespace ai::cv_detail
