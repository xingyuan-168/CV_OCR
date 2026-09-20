#include "cv_correlation.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

int main() {
    cv::setNumThreads(1);
    cv::RNG random(9721);
    for (auto shape : {cv::Size(610, 318), cv::Size(117, 93), cv::Size(64, 64), cv::Size(641, 49)}) {
        cv::Mat image(shape, CV_8UC3);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        for (auto size : {cv::Size(29, 26), cv::Size(27, 22), cv::Size(1, 1)}) {
            cv::Mat templ = image(cv::Rect(7, 9, size.width, size.height)).clone();
            const auto spectrum =
                ai::cv_detail::make_spectrum(templ, ai::cv_detail::spectrum_geometry(shape));
            cv::Scalar mean, sd;
            cv::meanStdDev(templ, mean, sd);
            double sums[3], energy = 0;
            for (int c = 0; c < 3; ++c) {
                sums[c] = mean[c] * templ.total();
                energy += sd[c] * sd[c] * templ.total();
            }
            ai::cv_detail::CorrelationWorkspace workspace;
            cv::Mat score, reference;
            workspace.prepare(image);
            for (bool centered : {true, false}) {
                workspace.match(templ, spectrum, sums, energy, centered, score);
                cv::matchTemplate(image, templ, reference,
                                  centered ? cv::TM_CCOEFF_NORMED : cv::TM_CCORR_NORMED);
                const double error = cv::norm(score, reference, cv::NORM_INF);
                std::cout << shape << " " << size << " centered=" << centered << " error=" << error << '\n';
                assert(error < 2e-5);
                cv::Mat filtered;
                workspace.match(templ, spectrum, sums, energy, centered, filtered, 0.399998f);
                for (int y = 0; y < score.rows; ++y)
                    for (int x = 0; x < score.cols; ++x) {
                        if (score.at<float>(y, x) >= 0.4f)
                            assert(filtered.at<float>(y, x) == score.at<float>(y, x));
                    }
                if (!centered) {
                    workspace.prepare(image, false);
                    workspace.match(templ, spectrum, sums, energy, false, filtered);
                    assert(cv::norm(score, filtered, cv::NORM_INF) < 2e-5);
                    workspace.prepare(image);
                }
            }
            if (shape.width == 610 && size.width == 29) {
                double totals[4]{};
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 100; ++i) {
                    image.at<cv::Vec3b>(0, 0)[0] = static_cast<uint8_t>(i);
                    workspace.prepare(image);
                    workspace.match(templ, spectrum, sums, energy, true, score);
                    for (int j = 0; j < 4; ++j)
                        totals[j] += workspace.timings[j];
                }
                for (double value : totals)
                    std::cout << value / 100 << " ";
                std::cout << "total="
                          << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                       start)
                                     .count() /
                                 100
                          << '\n';
            }
            workspace.discard();
            assert(workspace.retained_bytes() == 0);
        }
    }
    // Exercise the non-AVX2 fallback on this same host; optimized results must
    // remain independent of the availability of the private SIMD translation unit.
    cv::Mat flat(33, 641, CV_8UC3, cv::Scalar(127, 127, 127));
    cv::Mat templ(3, 5, CV_8UC3, cv::Scalar(127, 127, 127)), score;
    const double sums[3] = {1905, 1905, 1905};
    auto spectrum = ai::cv_detail::make_spectrum(templ, ai::cv_detail::spectrum_geometry(flat.size()));
    cv::setUseOptimized(false);
    ai::cv_detail::CorrelationWorkspace workspace;
    workspace.prepare(flat);
    workspace.match(templ, spectrum, sums, 0, true, score);
    assert(cv::countNonZero(score != 1) == 0);
    random.fill(flat, cv::RNG::UNIFORM, 0, 256);
    templ = flat(cv::Rect(11, 13, 19, 7)).clone();
    spectrum = ai::cv_detail::make_spectrum(templ, ai::cv_detail::spectrum_geometry(flat.size()));
    cv::Scalar mean, sd;
    cv::meanStdDev(templ, mean, sd);
    double varied_sums[3], energy = 0;
    for (int c = 0; c < 3; ++c) {
        varied_sums[c] = mean[c] * templ.total();
        energy += sd[c] * sd[c] * templ.total();
    }
    workspace.prepare(flat);
    workspace.match(templ, spectrum, varied_sums, energy, true, score);
    cv::Mat reference;
    cv::matchTemplate(flat, templ, reference, cv::TM_CCOEFF_NORMED);
    assert(cv::norm(score, reference, cv::NORM_INF) < 2e-5);
    cv::setUseOptimized(true);
}
