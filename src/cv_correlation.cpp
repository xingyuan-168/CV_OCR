#include "cv_correlation.h"
#include "cv_normalize.h"
#include "cv_masked_score_avx2.h"
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
MaskedTemplateSpectrum make_masked_spectrum(const cv::Mat& templ, const cv::Mat& mask,
                                           cv::Size geometry) {
    CV_Assert((templ.channels() == 1 || templ.channels() == 3) && templ.depth() == CV_8U &&
              mask.size() == templ.size() && mask.depth() == CV_8U);
    MaskedTemplateSpectrum result;
    result.geometry = geometry;
    result.extent = templ.size();
    result.channels = templ.channels();
    uint64_t sums[3]{}, squares[3]{};
    for (int y = 0; y < templ.rows; ++y)
        for (int x = 0; x < templ.cols; ++x) {
            if (!mask.ptr<uint8_t>(y)[x * mask.channels()]) continue;
            result.visible.push_back({x,y});
            for (int c = 0; c < result.channels; ++c) {
                const uint32_t v = templ.ptr<uint8_t>(y)[x * result.channels + c];
                sums[c] += v; squares[c] += v * v;
            }
        }
    result.count = result.visible.size();
    if (!result.count) return result;
    double channel_energy[3]{};
    for(int c=0;c<result.channels;++c){channel_energy[c]=double(squares[c])-double(sums[c])*sums[c]/result.count;result.energy+=channel_energy[c];result.energy_error+=16*DBL_EPSILON*(double(squares[c])+1);}
    double absolute_sum = 0;
    cv::Mat plane(geometry, CV_32F);
    for (int c = 0; c < result.channels; ++c) {
        const double mean = static_cast<double>(sums[c]) / result.count;
        const double energy = static_cast<double>(squares[c]) - mean * sums[c];
        if (energy <= 1e-12) continue;
        result.active_channels |= 1u << c;
        plane.setTo(0);
        for (auto p : result.visible) {
            const double v = templ.ptr<uint8_t>(p.y)[p.x * result.channels + c] - mean;
            absolute_sum += std::abs(v);
            plane.at<float>(p)=static_cast<float>(v);
        }
        cv::dft(plane, result.centered[c]);
    }
    // Conservative absolute guard for the float transform/multiply/inverse
    // chain, including coefficient rounding. Low-information/threshold-boundary
    // windows remain candidates rather than being rejected by a noisy estimate.
    const double stages = std::max(1.0, std::ceil(std::log2(static_cast<double>(geometry.area()))));
    const double epsilon=FLT_EPSILON;
    result.numerator_error = 64.0 * epsilon * stages * stages * 128.0 * absolute_sum +
        8.0 * epsilon * 255.0 * result.count;
    // Disjoint pairs spread over the support, not the filled bounding box.
    const size_t pair_count = std::min<size_t>(16, result.count / 2);
    for (size_t k = 0; k < pair_count; ++k) {
        const size_t a = k * result.count / (2 * pair_count);
        const size_t b = result.count - 1 - a;
        result.pairs.emplace_back(result.visible[a], result.visible[b]);
    }
    // Disjoint same-channel differences are orthogonal to each other and to
    // every per-channel DC component. Pair opposite extremes, then visit the
    // largest template-energy directions first. This needs only two integer
    // moments per candidate, without source means or a full-image transform.
    if (result.count <= 1024 && result.energy > 1e-12) {
        struct Pair{cv::Point a,b;int channel,difference;};std::vector<Pair> ordered;
        for(int c=0;c<result.channels;++c){auto points=result.visible;
            std::stable_sort(points.begin(),points.end(),[&](cv::Point a,cv::Point b){return templ.ptr<uint8_t>(a.y)[a.x*result.channels+c]<templ.ptr<uint8_t>(b.y)[b.x*result.channels+c];});
            for(size_t k=0;k<points.size()/2;++k){auto a=points[k],b=points[points.size()-1-k];
                int difference=int(templ.ptr<uint8_t>(b.y)[b.x*result.channels+c])-templ.ptr<uint8_t>(a.y)[a.x*result.channels+c];
                if(difference)ordered.push_back({a,b,c,difference});}
        }
        std::stable_sort(ordered.begin(),ordered.end(),[](const Pair&a,const Pair&b){return a.difference>b.difference;});
        double energy=0;
        for(size_t k=0;k<ordered.size();++k){auto p=ordered[k];result.projection_pairs.emplace_back(p.a,p.b);result.projection_channels.push_back(p.channel);result.projection_differences.push_back(p.difference);energy+=p.difference*p.difference;
            const size_t n=k+1;if((n&(n-1))==0||n==ordered.size()){MaskedProjectionStage stage;stage.end=n;stage.energy=energy;result.projection_stages.push_back(stage);}
        }
    }
    return result;
}

void MaskedCorrelationWorkspace::prepare(const cv::Mat& image, const MaskedTemplateSpectrum& templ) {
    if (geometry != templ.geometry || channels != image.channels()) {
        invalidate(); geometry = templ.geometry; channels = image.channels();
    }
    plane.create(geometry, CV_32F);
    for (int c = 0; c < channels; ++c) {
        if (!(templ.active_channels & (1u << c)) || (prepared_channels & (1u << c))) continue;
        plane.setTo(0);
        bool constant = true;
        const uint8_t first_value = image.ptr<uint8_t>(0)[c];
        for (int y = 0; y < image.rows; ++y) {
            const auto* src = image.ptr<uint8_t>(y);
            for (int x = 0; x < image.cols; ++x) {
                const uint8_t v=src[x * channels + c];
                plane.ptr<float>(y)[x]=float(int(v)-128);
                constant &= v==first_value;
            }
        }
        if (constant) constant_channels |= 1u << c;
        else fft.transform(plane, spectra[c], false);
        prepared_channels |= 1u << c;
    }
}

void MaskedCorrelationWorkspace::upper_bounds(const cv::Mat& image,
        const MaskedTemplateSpectrum& templ, float threshold, cv::Mat& output) {
    CV_Assert(templ.geometry.width >= image.cols && templ.geometry.height >= image.rows);
    if(!templ.projection_stages.empty() && threshold>1e-6f && image.total()<=static_cast<size_t>(INT32_MAX/3)) {
        // The public scorer rounds tofloat and snaps scores above1-1e-6 to1.
        // Also cover its double moment arithmetic, not just the mathematical
        // correlation. See docs/CV_MASKED_PROJECTION_NUMERICS.md.
        const int ch=image.channels();const double tau=std::max(0.,double(threshold)-1.5e-6);const double tau2=tau*tau;
        output.create(image.rows-templ.extent.height+1,image.cols-templ.extent.width+1,CV_32F);
        // Contiguous adjacent windows replace two expensive BGR gathers per
        // direction with two eight-byte loads. Only pixels, never match
        // results, are reused after the outer cache verifies frame bytes.
        if(!projection_prepared || projection_planes.cols!=image.cols || projection_planes.rows!=image.rows*ch){
            projection_planes.create(image.rows*ch,image.cols,CV_8U);
            if(ch==1)image.copyTo(projection_planes);
            else {cv::Mat planes[3];for(int c=0;c<3;++c)planes[c]=projection_planes.rowRange(c*image.rows,(c+1)*image.rows);cv::split(image,planes);}
            projection_constant=true;
            for(int c=0;c<ch&&projection_constant;++c){const auto*first=projection_planes.ptr<uint8_t>(c*image.rows);
                for(int x=1;x<image.cols&&projection_constant;++x)projection_constant=first[x]==first[0];
                for(int y=1;y<image.rows&&projection_constant;++y)projection_constant=std::memcmp(first,projection_planes.ptr<uint8_t>(c*image.rows+y),image.cols)==0;}
            projection_prepared=true;
        }
        if(projection_constant){output.setTo(-1);return;}
        std::array<MaskedProjectionLimit,16> limits{};size_t limit_count=0;
        for(const auto&stage:templ.projection_stages){const double d=stage.energy-(1-tau2)*2*templ.energy;
            const double guard=2*templ.energy_error+64*DBL_EPSILON*(stage.energy+templ.energy+1);
            if(d>guard){limits[limit_count++]={stage.end,d-guard,static_cast<float>((d-guard)*(1-32*FLT_EPSILON))};}}
        if(!limit_count){output.setTo(1);return;}
        projection_offsets.resize(templ.projection_pairs.size()*2);
        for(size_t k=0;k<templ.projection_pairs.size();++k){auto p=templ.projection_pairs[k];int c=templ.projection_channels[k];projection_offsets[2*k]=uint32_t((c*image.rows+p.first.y)*projection_planes.step+p.first.x);projection_offsets[2*k+1]=uint32_t((c*image.rows+p.second.y)*projection_planes.step+p.second.x);}
        for(int y=0;y<output.rows;++y){auto*dst=output.ptr<float>(y);int first=0;
#if defined(AIENGINE_CV_AVX2)
            if(cv::checkHardwareSupport(CV_CPU_AVX2))first=masked_projection_candidates_avx2(projection_planes.ptr<uint8_t>(y),projection_offsets.data(),templ.projection_differences.data(),limits.data(),limit_count,output.cols,dst);
#endif
            for(int x=first;x<output.cols;++x){int64_t dot=0;uint64_t q=0;size_t k=0;bool excluded=false;
                const auto*base=projection_planes.ptr<uint8_t>(y)+x;
                for(size_t level=0;level<limit_count;++level){const auto&stage=limits[level];
                    for(;k<stage.end;++k){const int v=int(base[projection_offsets[2*k+1]])-base[projection_offsets[2*k]];q+=v*v;dot+=v*templ.projection_differences[k];}
                    if(q==0){excluded=true;break;}
                    const double upper=std::max(0.,double(dot));
                    if(upper*upper*(1+16*DBL_EPSILON)<double(q)*stage.exact*(1-16*DBL_EPSILON)){excluded=true;break;}
                }dst[x]=excluded?-1.f:1.f;
            }
        }return;
    }
    prepare(image, templ);
    output.create(image.rows - templ.extent.height + 1, image.cols - templ.extent.width + 1, CV_32F);
    if (!templ.active_channels) { output.setTo(1); numerator.release(); return; }
    if ((constant_channels & templ.active_channels) == templ.active_channels &&
        threshold>1e-6f) {
        // Every active channel is constant over the whole image. Its centered
        // dot product is exactly zero for every mask/window, not an FFT estimate.
        output.setTo(-1); return;
    }
    bool first = true;
    for (int c = 0; c < channels; ++c) if ((templ.active_channels & (1u << c)) && !(constant_channels & (1u << c))) {
        cv::mulSpectrums(spectra[c], templ.centered[c], product, 0, true);
        if (first) { product.copyTo(accumulated); first = false; }
        else cv::add(accumulated, product, accumulated);
    }
    if(first){numerator.create(templ.geometry,CV_32F);numerator.setTo(0);}
    else fft.transform(accumulated, numerator, true);
    const double effective = std::max(0.0, static_cast<double>(threshold) - 1e-6);
    std::vector<uint32_t> pair_offsets;
    for(const auto& p:templ.pairs){pair_offsets.push_back(static_cast<uint32_t>(p.first.y*image.step+p.first.x*channels));pair_offsets.push_back(static_cast<uint32_t>(p.second.y*image.step+p.second.x*channels));}
    const double budget=effective*effective*templ.energy;
    if(budget<=0){output.setTo(1);return;}
    const double reject = budget;
    for (int y = 0; y < output.rows; ++y) {
        auto* dst = output.ptr<float>(y);
        const auto* num = numerator.ptr<float>(y);
        int first_x=0;
#if defined(AIENGINE_CV_AVX2)
        if(cv::checkHardwareSupport(CV_CPU_AVX2)&&image.total()<=static_cast<size_t>(INT32_MAX/3))
            first_x=masked_pair_candidates_avx2(image.ptr<uint8_t>(y),channels,(1u<<channels)-1,pair_offsets.data(),templ.pairs.size(),num,
                templ.numerator_error,reject,output.cols,dst);
#endif
        for (int x = first_x; x < output.cols; ++x) {
            const double upper_num = num[x] + templ.numerator_error;
            if (!std::isfinite(upper_num)) { dst[x] = 1; continue; }
            if (upper_num < 0) { dst[x] = -1; continue; }
            uint32_t subset_sums[3]{},subset_square_sum=0;size_t used=0;
            bool excluded = false;
            for (const auto& pair : templ.pairs) {
                const auto* a = image.ptr<uint8_t>(y + pair.first.y) + (x + pair.first.x) * channels;
                const auto* b = image.ptr<uint8_t>(y + pair.second.y) + (x + pair.second.x) * channels;
                for (int c = 0; c < channels; ++c) {
                    const uint32_t va=a[c],vb=b[c];subset_sums[c]+=va+vb;
                    subset_square_sum+=va*va+vb*vb;
                }
                used+=2;double lower=subset_square_sum;
                for(int c=0;c<channels;++c)lower-=double(subset_sums[c])*subset_sums[c]/used;
                if (upper_num * upper_num < reject * std::max(0.,lower)) { excluded = true; break; }
            }
            if (excluded) { dst[x] = -1; continue; }
            dst[x] = 1.0f;
        }
    }
}

void MaskedCorrelationWorkspace::discard() noexcept {
    std::vector<uint32_t>().swap(projection_offsets);
    projection_planes.release();projection_prepared=false;
    fft.discard();
    for (auto& s : spectra) s.release();
    plane.release(); product.release(); accumulated.release(); numerator.release();
    prepared_channels = constant_channels = 0; geometry = {}; channels = 0;
}

size_t MaskedCorrelationWorkspace::retained_bytes() const noexcept {
    size_t result = bytes(projection_planes) + projection_offsets.capacity()*sizeof(uint32_t) + fft.retained_bytes() + bytes(plane) + bytes(product) + bytes(accumulated) + bytes(numerator);
    for (const auto& s : spectra) result += bytes(s);
    return result;
}

} // namespace ai::cv_detail
