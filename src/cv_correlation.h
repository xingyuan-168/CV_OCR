#pragma once

#include <array>
#include <opencv2/core.hpp>

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

} // namespace ai::cv_detail
