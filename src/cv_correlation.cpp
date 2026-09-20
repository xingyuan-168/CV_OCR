#include "cv_correlation.h"
#include "cv_normalize.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>
#if defined(AIENGINE_CV_IPP_PLAN)
#include <ippicv.h>
#endif

namespace ai::cv_detail {
float CorrelationWorkspace::exact_at(const cv::Mat &image, const cv::Mat &templ, const double *template_sums,
                                     double template_energy, int x, int y) const {
    if (template_energy <= 1e-12)
        return -1;
    const double inv = 1.0 / (static_cast<double>(templ.rows) * templ.cols);
    const auto window = [&](const cv::Mat &m) {
        const auto *top = m.ptr<double>(y);
        const auto *bottom = m.ptr<double>(y + templ.rows);
        return top[x] - top[x + templ.cols] - bottom[x] + bottom[x + templ.cols];
    };
    double energy = window(square_sum);
    double dot = image(cv::Rect(x, y, templ.cols, templ.rows)).dot(templ);
    for (int c = 0; c < 3; ++c) {
        const auto *top = sums[c].ptr<int>(y);
        const auto *bottom = sums[c].ptr<int>(y + templ.rows);
        const double value = top[x] - top[x + templ.cols] - bottom[x] + bottom[x + templ.cols];
        dot -= value * template_sums[c] * inv;
        energy -= value * value * inv;
    }
    if (energy <= 1e-12)
        return -1;
    const double result = dot / std::sqrt(energy * template_energy);
    if (!std::isfinite(result))
        return -1;
    return result > 1 - 1e-6 ? 1.0f : static_cast<float>(std::max(-1.0, std::min(1.0, result)));
}
namespace {
size_t bytes(const cv::Mat &m) noexcept {
    return m.data ? static_cast<size_t>(m.datalimit - m.datastart) : 0;
}
} // namespace

cv::Size spectrum_geometry(cv::Size image) {
    // Only valid correlations are read, so padding to image size is sufficient:
    // no valid window wraps around either edge of the circular correlation.
    return {cv::getOptimalDFTSize((image.width + 31) & ~31),
            cv::getOptimalDFTSize((image.height + 31) & ~31)};
}

TemplateSpectrum make_spectrum(const cv::Mat &templ, cv::Size geometry) {
    TemplateSpectrum result;
    result.geometry = geometry;
    cv::Mat plane(geometry, CV_32F);
    for (int c = 0; c < 3; ++c) {
        plane.setTo(0);
        for (int y = 0; y < templ.rows; ++y) {
            const auto *in = templ.ptr<uint8_t>(y);
            auto *out = plane.ptr<float>(y);
            for (int x = 0; x < templ.cols; ++x)
                out[x] = in[x * 3 + c];
        }
        cv::dft(plane, result.channels[c]);
    }
    return result;
}

void CorrelationWorkspace::discard() noexcept {
    for (auto &m : planes)
        m.release();
    for (auto &m : spectra)
        m.release();
    for (auto &m : sums)
        m.release();
    square_sum.release();
    product.release();
    accumulated.release();
    correlation.release();
    transform_plan.release();
    transform_buffer.release();
    plan_geometry = {};
}

size_t CorrelationWorkspace::retained_bytes() const noexcept {
    size_t result = bytes(transform_plan) + bytes(transform_buffer) + bytes(square_sum) + bytes(product) +
                    bytes(accumulated) + bytes(correlation);
    for (const auto &m : planes)
        result += bytes(m);
    for (const auto &m : spectra)
        result += bytes(m);
    for (const auto &m : sums)
        result += bytes(m);
    return result;
}

void CorrelationWorkspace::transform(const cv::Mat &input, cv::Mat &output, bool inverse) {
#if defined(AIENGINE_CV_IPP_PLAN)
    // cv::dft creates and frees the IPP plan on every call. Each leased workspace
    // owns one plan and work buffer; they are counted in the idle cache budget.
    if (plan_geometry != input.size()) {
        int spec_size = 0, init_size = 0, buffer_size = 0;
        const IppiSize size{input.cols, input.rows};
        auto status = ippiDFTGetSize_R_32f(size, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &spec_size, &init_size,
                                           &buffer_size);
        if (status < 0)
            CV_Error(cv::Error::StsError, "IPP DFT size failed");
        transform_plan.create(1, std::max(1, spec_size), CV_8U);
        transform_buffer.create(1, std::max(1, buffer_size), CV_8U);
        cv::Mat init(1, std::max(1, init_size), CV_8U);
        status = ippiDFTInit_R_32f(size, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone,
                                   reinterpret_cast<IppiDFTSpec_R_32f *>(transform_plan.data), init.data);
        if (status < 0)
            CV_Error(cv::Error::StsError, "IPP DFT initialization failed");
        plan_geometry = input.size();
    }
    output.create(input.size(), CV_32F);
    const auto *spec = reinterpret_cast<const IppiDFTSpec_R_32f *>(transform_plan.data);
    const auto status = inverse
                            ? ippiDFTInv_PackToR_32f_C1R(input.ptr<float>(), static_cast<int>(input.step),
                                                         output.ptr<float>(), static_cast<int>(output.step),
                                                         spec, transform_buffer.data)
                            : ippiDFTFwd_RToPack_32f_C1R(input.ptr<float>(), static_cast<int>(input.step),
                                                         output.ptr<float>(), static_cast<int>(output.step),
                                                         spec, transform_buffer.data);
    if (status < 0)
        CV_Error(cv::Error::StsError, "IPP DFT execution failed");
#else
    cv::dft(input, output, inverse ? cv::DFT_INVERSE | cv::DFT_SCALE : 0);
#endif
}

void CorrelationWorkspace::prepare(const cv::Mat &image, bool centered) {
    const auto start = std::chrono::steady_clock::now();
    const auto geometry = spectrum_geometry(image.size());
    for (auto &plane : planes) {
        plane.create(geometry, CV_32F);
        plane.setTo(0);
    }
    // The fast path is limited to 512K pixels, so channel sums are at most
    // 133693440. Int32 integrals are exact and halve their memory traffic.
    CV_Assert(image.total() <= 512u * 1024u);
    for (auto &sum : sums) {
        sum.create(image.rows + 1, image.cols + 1, CV_32S);
        sum.row(0).setTo(0);
    }
    square_sum.create(image.rows + 1, image.cols + 1, CV_64F);
    square_sum.row(0).setTo(0);
    for (int y = 0; y < image.rows; ++y) {
        const auto *in = image.ptr<uint8_t>(y);
        auto *b = planes[0].ptr<float>(y);
        auto *g = planes[1].ptr<float>(y);
        auto *r = planes[2].ptr<float>(y);
        int *sb = sums[0].ptr<int>(y + 1), *sg = sums[1].ptr<int>(y + 1), *sr = sums[2].ptr<int>(y + 1);
        const int *pb = sums[0].ptr<int>(y), *pg = sums[1].ptr<int>(y), *pr = sums[2].ptr<int>(y);
        double *sq = square_sum.ptr<double>(y + 1);
        const double *pq = square_sum.ptr<double>(y);
        sb[0] = sg[0] = sr[0] = 0;
        sq[0] = 0;
        double carry[4]{};
        int x = 0;
#if defined(AIENGINE_CV_AVX2)
        if (cv::checkHardwareSupport(CV_CPU_AVX2)) {
            float *p[3] = {b, g, r};
            int *s[3] = {sb, sg, sr};
            const int *prev[3] = {pb, pg, pr};
            x = prepare_row_avx2(in, image.cols, p, s, prev, sq, pq, carry, centered);
        }
#endif
        int vb = static_cast<int>(carry[0]), vg = static_cast<int>(carry[1]), vr = static_cast<int>(carry[2]);
        double vq = carry[3];
        for (; x < image.cols; ++x) {
            const int ib = in[x * 3], ig = in[x * 3 + 1], ir = in[x * 3 + 2];
            b[x] = static_cast<float>(ib);
            g[x] = static_cast<float>(ig);
            r[x] = static_cast<float>(ir);
            vb += ib;
            vg += ig;
            vr += ir;
            vq += ib * ib + ig * ig + ir * ir;
            if (centered) {
                sb[x + 1] = pb[x + 1] + vb;
                sg[x + 1] = pg[x + 1] + vg;
                sr[x + 1] = pr[x + 1] + vr;
            }
            sq[x + 1] = pq[x + 1] + vq;
        }
    }
    const auto integrated = std::chrono::steady_clock::now();
    // nonzeroRows disables OpenCV's two-dimensional IPP implementation, even
    // for an almost full frame. The padded rows are explicitly zeroed above.
    for (int c = 0; c < 3; ++c)
        transform(planes[c], spectra[c], false);
    const auto transformed = std::chrono::steady_clock::now();
    timings[0] = std::chrono::duration<double, std::milli>(transformed - integrated).count();
    timings[1] = std::chrono::duration<double, std::milli>(integrated - start).count();
}

void CorrelationWorkspace::match(const cv::Mat &templ, const TemplateSpectrum &spectrum, const double *sums,
                                 double energy, bool centered, cv::Mat &score, float floor) {
    const auto start = std::chrono::steady_clock::now();
    bool summed = false;
#if defined(AIENGINE_CV_AVX2)
    if (cv::checkHardwareSupport(CV_CPU_AVX2)) {
        accumulated.create(spectrum.geometry, CV_32F);
        const float *a[3] = {spectra[0].ptr<float>(), spectra[1].ptr<float>(), spectra[2].ptr<float>()};
        const float *b[3] = {spectrum.channels[0].ptr<float>(), spectrum.channels[1].ptr<float>(),
                             spectrum.channels[2].ptr<float>()};
        spectrum_sum_avx2(a, b, accumulated.ptr<float>(), accumulated.rows, accumulated.cols);
        summed = true;
    }
#endif
    if (!summed) {
        cv::mulSpectrums(spectra[0], spectrum.channels[0], accumulated, 0, true);
        for (int c = 1; c < 3; ++c) {
            cv::mulSpectrums(spectra[c], spectrum.channels[c], product, 0, true);
            cv::add(accumulated, product, accumulated);
        }
    }
    const int rows = square_sum.rows - templ.rows, cols = square_sum.cols - templ.cols;
    transform(accumulated, correlation, true);
    const auto transformed = std::chrono::steady_clock::now();
    score.create(rows, cols, CV_32F);
    const double inv_area = 1.0 / (static_cast<double>(templ.rows) * templ.cols);
    if (centered && energy * inv_area < DBL_EPSILON) {
        score.setTo(1);
        return;
    }
    double norm2 = energy;
    if (!centered)
        for (int c = 0; c < 3; ++c)
            norm2 += sums[c] * sums[c] * inv_area;
    const double norm = std::sqrt(norm2);
    for (int y = 0; y < rows; ++y) {
        const int *s0[3] = {this->sums[0].ptr<int>(y), this->sums[1].ptr<int>(y), this->sums[2].ptr<int>(y)};
        const int *s1[3] = {this->sums[0].ptr<int>(y + templ.rows), this->sums[1].ptr<int>(y + templ.rows),
                            this->sums[2].ptr<int>(y + templ.rows)};
        const auto *q0 = square_sum.ptr<double>(y);
        const auto *q1 = square_sum.ptr<double>(y + templ.rows);
        const auto *raw = correlation.ptr<float>(y);
        auto *out = score.ptr<float>(y);
        int x = 0;
#if defined(AIENGINE_CV_AVX2)
        if (cv::checkHardwareSupport(CV_CPU_AVX2))
            x = normalize_avx2(s0, s1, q0, q1, raw, out, cols, templ.cols, sums, inv_area, norm, centered,
                               floor);
#endif
        for (; x < cols; ++x) {
            double num = raw[x], mean2 = 0, square = 0;
            const int right = x + templ.cols;
            if (centered)
                for (int c = 0; c < 3; ++c) {
                    const double value = s0[c][x] - s0[c][right] - s1[c][x] + s1[c][right];
                    num -= value * sums[c] * inv_area;
                    mean2 += value * value;
                }
            square = q0[x] - q0[right] - q1[x] + q1[right];
            const double variance = std::max(0.0, square - mean2 * inv_area);
            if (floor > 0 &&
                (num <= 0 || num * num < static_cast<double>(floor) * floor * variance * norm2)) {
                out[x] = -1;
                continue;
            }
            const double denominator =
                variance <= std::min(0.5, 10 * FLT_EPSILON * square) ? 0 : std::sqrt(variance) * norm;
            if (std::abs(num) < denominator)
                num /= denominator;
            else if (std::abs(num) < denominator * 1.125)
                num = num > 0 ? 1 : -1;
            else
                num = 0;
            out[x] = static_cast<float>(num);
        }
    }
    timings[2] = std::chrono::duration<double, std::milli>(transformed - start).count();
    timings[3] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - transformed).count();
}
} // namespace ai::cv_detail
