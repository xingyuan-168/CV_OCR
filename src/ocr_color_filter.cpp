#include "ocr_color_filter.h"

#include "backends.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>

namespace ai {
namespace {

constexpr size_t kMaxRules = 16;
constexpr size_t kMaxFilterBytes = 512;
constexpr int32_t kMaxScaledSide = 1536;
constexpr int64_t kMaxScaledPixels = 4000000;
constexpr float kPreferredScale = 3.0f;
constexpr size_t kMaxExecutedCandidates = 5;

void summarize_foreground(OcrFilterCandidate* candidate);
OcrFilterCandidate binary_from_mask(
    const OcrFilterCandidate& gray,
    const std::vector<uint8_t>& mask,
    const char* mode);

bool is_hex(char value) {
    return (value >= '0' && value <= '9') ||
        (value >= 'a' && value <= 'f') ||
        (value >= 'A' && value <= 'F');
}

uint8_t hex_pair(const char* value) {
    const auto digit = [](char ch) -> uint8_t {
        if (ch >= '0' && ch <= '9') return static_cast<uint8_t>(ch - '0');
        if (ch >= 'a' && ch <= 'f') return static_cast<uint8_t>(ch - 'a' + 10);
        return static_cast<uint8_t>(ch - 'A' + 10);
    };
    return static_cast<uint8_t>((digit(value[0]) << 4) | digit(value[1]));
}

std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(begin, end - begin);
}

bool source_pixel_bgr(
    const AIImage& source,
    int32_t x,
    int32_t y,
    uint8_t* b,
    uint8_t* g,
    uint8_t* r) {
    if (b == nullptr || g == nullptr || r == nullptr || source.data == nullptr ||
        x < 0 || y < 0 || x >= source.width || y >= source.height) return false;
    const uint8_t* row = source.data + static_cast<ptrdiff_t>(y) * source.stride;
    switch (source.format) {
    case AI_IMAGE_GRAY8:
        *b = *g = *r = row[x];
        return true;
    case AI_IMAGE_BGR24: {
        const uint8_t* p = row + static_cast<size_t>(x) * 3u;
        *b = p[0]; *g = p[1]; *r = p[2];
        return true;
    }
    case AI_IMAGE_BGRA32: {
        const uint8_t* p = row + static_cast<size_t>(x) * 4u;
        *b = p[0]; *g = p[1]; *r = p[2];
        return true;
    }
    case AI_IMAGE_RGB24: {
        const uint8_t* p = row + static_cast<size_t>(x) * 3u;
        *r = p[0]; *g = p[1]; *b = p[2];
        return true;
    }
    case AI_IMAGE_RGBA32: {
        const uint8_t* p = row + static_cast<size_t>(x) * 4u;
        *r = p[0]; *g = p[1]; *b = p[2];
        return true;
    }
    default:
        return false;
    }
}

void write_gray(
    std::vector<uint8_t>* pixels,
    int32_t width,
    int32_t x,
    int32_t y,
    uint8_t value) {
    uint8_t* p = pixels->data() +
        (static_cast<size_t>(y) * static_cast<size_t>(width) +
         static_cast<size_t>(x)) * 3u;
    p[0] = p[1] = p[2] = value;
}

OcrFilterCandidate copy_bgr(const AIImage& source) {
    OcrFilterCandidate output;
    output.width = source.width;
    output.height = source.height;
    output.stride = source.width * 3;
    output.mode = "original";
    output.pixels.resize(
        static_cast<size_t>(output.stride) * static_cast<size_t>(output.height));
    for (int32_t y = 0; y < source.height; ++y) {
        for (int32_t x = 0; x < source.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            source_pixel_bgr(source, x, y, &b, &g, &r);
            uint8_t* p = output.pixels.data() +
                (static_cast<size_t>(y) * static_cast<size_t>(output.width) +
                 static_cast<size_t>(x)) * 3u;
            p[0] = b;
            p[1] = g;
            p[2] = r;
        }
    }
    return output;
}

float adaptive_scale(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) return 1.0f;
    float scale = kPreferredScale;
    const int32_t longest = std::max(width, height);
    scale = std::min(scale, static_cast<float>(kMaxScaledSide) / longest);
    const double pixel_limit = std::sqrt(
        static_cast<double>(kMaxScaledPixels) /
        (static_cast<double>(width) * static_cast<double>(height)));
    scale = std::min(scale, static_cast<float>(pixel_limit));
    return std::max(1.0f, scale);
}

OcrFilterCandidate resize_candidate(
    const OcrFilterCandidate& input,
    float requested_scale,
    bool nearest,
    const char* mode) {
    if (requested_scale <= 1.0001f) {
        OcrFilterCandidate output = input;
        output.mode = mode;
        return output;
    }
    OcrFilterCandidate output;
    output.width = std::max<int32_t>(
        1, static_cast<int32_t>(std::lround(input.width * requested_scale)));
    output.height = std::max<int32_t>(
        1, static_cast<int32_t>(std::lround(input.height * requested_scale)));
    output.stride = output.width * 3;
    output.scale_x = static_cast<float>(output.width) / input.width;
    output.scale_y = static_cast<float>(output.height) / input.height;
    output.mode = mode;
    output.pixels.resize(
        static_cast<size_t>(output.stride) * static_cast<size_t>(output.height));

    for (int32_t y = 0; y < output.height; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) / output.scale_y - 0.5f;
        const int32_t y0 = std::clamp(
            static_cast<int32_t>(std::floor(sy)), 0, input.height - 1);
        const int32_t y1 = std::min(y0 + 1, input.height - 1);
        const float fy = std::clamp(sy - std::floor(sy), 0.0f, 1.0f);
        for (int32_t x = 0; x < output.width; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) / output.scale_x - 0.5f;
            const int32_t x0 = std::clamp(
                static_cast<int32_t>(std::floor(sx)), 0, input.width - 1);
            const int32_t x1 = std::min(x0 + 1, input.width - 1);
            const float fx = std::clamp(sx - std::floor(sx), 0.0f, 1.0f);
            const int32_t nx = std::clamp(
                static_cast<int32_t>(std::lround(sx)), 0, input.width - 1);
            const int32_t ny = std::clamp(
                static_cast<int32_t>(std::lround(sy)), 0, input.height - 1);
            uint8_t* dst = output.pixels.data() +
                (static_cast<size_t>(y) * output.width + x) * 3u;
            if (nearest) {
                const uint8_t* src = input.pixels.data() +
                    (static_cast<size_t>(ny) * input.width + nx) * 3u;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
                continue;
            }
            const uint8_t* p00 = input.pixels.data() +
                (static_cast<size_t>(y0) * input.width + x0) * 3u;
            const uint8_t* p10 = input.pixels.data() +
                (static_cast<size_t>(y0) * input.width + x1) * 3u;
            const uint8_t* p01 = input.pixels.data() +
                (static_cast<size_t>(y1) * input.width + x0) * 3u;
            const uint8_t* p11 = input.pixels.data() +
                (static_cast<size_t>(y1) * input.width + x1) * 3u;
            for (int c = 0; c < 3; ++c) {
                const float top = p00[c] * (1.0f - fx) + p10[c] * fx;
                const float bottom = p01[c] * (1.0f - fx) + p11[c] * fx;
                dst[c] = static_cast<uint8_t>(std::clamp(
                    top * (1.0f - fy) + bottom * fy, 0.0f, 255.0f));
            }
        }
    }
    if (!input.foreground_mask.empty()) {
        output.foreground_mask.assign(
            static_cast<size_t>(output.width) * output.height, 0);
        for (int32_t y = 0; y < output.height; ++y) {
            const int32_t source_y = std::clamp(
                static_cast<int32_t>(std::floor(y / output.scale_y)),
                0, input.height - 1);
            for (int32_t x = 0; x < output.width; ++x) {
                const int32_t source_x = std::clamp(
                    static_cast<int32_t>(std::floor(x / output.scale_x)),
                    0, input.width - 1);
                output.foreground_mask[static_cast<size_t>(y) * output.width + x] =
                    input.foreground_mask[
                        static_cast<size_t>(source_y) * input.width + source_x];
            }
        }
        summarize_foreground(&output);
    }
    return output;
}

void summarize_foreground(OcrFilterCandidate* candidate) {
    if (candidate == nullptr || candidate->width <= 0 || candidate->height <= 0 ||
        candidate->foreground_mask.size() !=
            static_cast<size_t>(candidate->width) * candidate->height) return;
    candidate->foreground_pixels = 0;
    candidate->active_columns = 0;
    candidate->foreground_x0 = candidate->width;
    candidate->foreground_y0 = candidate->height;
    candidate->foreground_x1 = 0;
    candidate->foreground_y1 = 0;
    for (int32_t x = 0; x < candidate->width; ++x) {
        bool active = false;
        for (int32_t y = 0; y < candidate->height; ++y) {
            if (candidate->foreground_mask[
                    static_cast<size_t>(y) * candidate->width + x] == 0) continue;
            active = true;
            ++candidate->foreground_pixels;
            candidate->foreground_x0 = std::min(candidate->foreground_x0, x);
            candidate->foreground_y0 = std::min(candidate->foreground_y0, y);
            candidate->foreground_x1 = std::max(candidate->foreground_x1, x + 1);
            candidate->foreground_y1 = std::max(candidate->foreground_y1, y + 1);
        }
        if (active) ++candidate->active_columns;
    }
    if (candidate->foreground_pixels == 0) {
        candidate->foreground_x0 = candidate->foreground_y0 = 0;
        candidate->foreground_x1 = candidate->foreground_y1 = 0;
    }
}

OcrFilterCandidate gray_contrast(const OcrFilterCandidate& input) {
    OcrFilterCandidate output = input;
    output.mode = "gray_contrast_3x";
    std::array<size_t, 256> histogram{};
    std::vector<uint8_t> gray(
        static_cast<size_t>(input.width) * static_cast<size_t>(input.height));
    for (size_t i = 0; i < gray.size(); ++i) {
        const uint8_t* p = input.pixels.data() + i * 3u;
        const uint8_t value = static_cast<uint8_t>(
            (static_cast<int>(p[2]) * 77 + static_cast<int>(p[1]) * 150 +
             static_cast<int>(p[0]) * 29 + 128) >> 8);
        gray[i] = value;
        ++histogram[value];
    }
    const size_t low_rank = gray.size() / 100;
    const size_t high_rank = gray.size() - gray.size() / 100;
    size_t cumulative = 0;
    int low = 0;
    int high = 255;
    for (int i = 0; i < 256; ++i) {
        cumulative += histogram[static_cast<size_t>(i)];
        if (cumulative >= low_rank) { low = i; break; }
    }
    cumulative = 0;
    for (int i = 255; i >= 0; --i) {
        cumulative += histogram[static_cast<size_t>(i)];
        if (cumulative >= gray.size() - high_rank) { high = i; break; }
    }
    if (high <= low) { low = 0; high = 255; }
    for (int32_t y = 0; y < output.height; ++y) {
        for (int32_t x = 0; x < output.width; ++x) {
            const size_t index = static_cast<size_t>(y) * output.width + x;
            const int stretched = (static_cast<int>(gray[index]) - low) * 255 /
                std::max(1, high - low);
            write_gray(
                &output.pixels,
                output.width,
                x,
                y,
                static_cast<uint8_t>(std::clamp(stretched, 0, 255)));
        }
    }
    return output;
}

OcrFilterCandidate gray_plain(const OcrFilterCandidate& input) {
    OcrFilterCandidate output = input;
    output.mode = "gray_plain";
    for (int32_t y = 0; y < output.height; ++y) {
        for (int32_t x = 0; x < output.width; ++x) {
            const size_t index = static_cast<size_t>(y) * output.width + x;
            const uint8_t* p = input.pixels.data() + index * 3u;
            const uint8_t value = static_cast<uint8_t>(
                (static_cast<int>(p[2]) * 77 + static_cast<int>(p[1]) * 150 +
                 static_cast<int>(p[0]) * 29 + 128) >> 8);
            write_gray(&output.pixels, output.width, x, y, value);
        }
    }
    return output;
}

bool useful_foreground(size_t count, size_t total) {
    return count > 0 && count * 1000 >= total && count * 100 <= total * 45;
}

bool choose_sparse_polarity(
    const std::vector<uint8_t>& bright,
    const std::vector<uint8_t>& dark,
    size_t bright_count,
    size_t dark_count,
    std::vector<uint8_t>* chosen) {
    if (chosen == nullptr || bright.size() != dark.size()) return false;
    const size_t total = bright.size();
    const bool bright_ok = useful_foreground(bright_count, total);
    const bool dark_ok = useful_foreground(dark_count, total);
    if (!bright_ok && !dark_ok) return false;
    if (bright_ok && !dark_ok) { *chosen = bright; return true; }
    if (!bright_ok && dark_ok) { *chosen = dark; return true; }
    const size_t target = total * 8 / 100;
    const size_t bright_distance = bright_count > target ? bright_count - target : target - bright_count;
    const size_t dark_distance = dark_count > target ? dark_count - target : target - dark_count;
    *chosen = bright_distance <= dark_distance ? bright : dark;
    return true;
}

OcrFilterCandidate binary_from_mask(
    const OcrFilterCandidate& gray,
    const std::vector<uint8_t>& mask,
    const char* mode) {
    OcrFilterCandidate output = gray;
    output.mode = mode;
    for (int32_t y = 0; y < output.height; ++y) {
        for (int32_t x = 0; x < output.width; ++x) {
            const size_t index = static_cast<size_t>(y) * output.width + x;
            write_gray(&output.pixels, output.width, x, y, mask[index] ? 255 : 0);
        }
    }
    output.foreground_mask = mask;
    summarize_foreground(&output);
    return output;
}

std::vector<OcrFilterCandidate> make_otsu_candidates(
    const OcrFilterCandidate& gray) {
    std::vector<OcrFilterCandidate> output;
    std::array<size_t, 256> histogram{};
    const size_t total = static_cast<size_t>(gray.width) * gray.height;
    for (size_t i = 0; i < total; ++i) ++histogram[gray.pixels[i * 3u]];
    double total_sum = 0.0;
    for (int i = 0; i < 256; ++i) total_sum += i * histogram[static_cast<size_t>(i)];
    size_t background_count = 0;
    double background_sum = 0.0;
    double best_variance = -1.0;
    int threshold = 127;
    for (int i = 0; i < 256; ++i) {
        background_count += histogram[static_cast<size_t>(i)];
        if (background_count == 0 || background_count == total) continue;
        background_sum += i * histogram[static_cast<size_t>(i)];
        const size_t foreground_count = total - background_count;
        const double mean_background = background_sum / background_count;
        const double mean_foreground = (total_sum - background_sum) / foreground_count;
        const double delta = mean_background - mean_foreground;
        const double variance = static_cast<double>(background_count) *
            foreground_count * delta * delta;
        if (variance > best_variance) {
            best_variance = variance;
            threshold = i;
        }
    }
    std::vector<uint8_t> bright(total, 0);
    std::vector<uint8_t> dark(total, 0);
    size_t bright_count = 0;
    size_t dark_count = 0;
    for (size_t i = 0; i < total; ++i) {
        const uint8_t value = gray.pixels[i * 3u];
        if (value > threshold) { bright[i] = 1; ++bright_count; }
        if (value <= threshold) { dark[i] = 1; ++dark_count; }
    }
    if (useful_foreground(bright_count, total)) {
        output.push_back(binary_from_mask(gray, bright, "otsu_bright_3x"));
    }
    if (useful_foreground(dark_count, total)) {
        output.push_back(binary_from_mask(gray, dark, "otsu_dark_3x"));
    }
    return output;
}

bool make_adaptive_candidate(
    const OcrFilterCandidate& gray,
    OcrFilterCandidate* output) {
    if (output == nullptr) return false;
    const int32_t width = gray.width;
    const int32_t height = gray.height;
    const size_t total = static_cast<size_t>(width) * height;
    std::vector<uint64_t> integral(
        static_cast<size_t>(width + 1) * static_cast<size_t>(height + 1), 0);
    for (int32_t y = 0; y < height; ++y) {
        uint64_t row_sum = 0;
        for (int32_t x = 0; x < width; ++x) {
            row_sum += gray.pixels[(static_cast<size_t>(y) * width + x) * 3u];
            integral[static_cast<size_t>(y + 1) * (width + 1) + x + 1] =
                integral[static_cast<size_t>(y) * (width + 1) + x + 1] + row_sum;
        }
    }
    std::vector<uint8_t> bright(total, 0);
    std::vector<uint8_t> dark(total, 0);
    size_t bright_count = 0;
    size_t dark_count = 0;
    constexpr int radius = 15;
    constexpr int offset = 8;
    for (int32_t y = 0; y < height; ++y) {
        const int32_t y0 = std::max(0, y - radius);
        const int32_t y1 = std::min(height, y + radius + 1);
        for (int32_t x = 0; x < width; ++x) {
            const int32_t x0 = std::max(0, x - radius);
            const int32_t x1 = std::min(width, x + radius + 1);
            const uint64_t sum =
                integral[static_cast<size_t>(y1) * (width + 1) + x1] -
                integral[static_cast<size_t>(y0) * (width + 1) + x1] -
                integral[static_cast<size_t>(y1) * (width + 1) + x0] +
                integral[static_cast<size_t>(y0) * (width + 1) + x0];
            const int mean = static_cast<int>(sum /
                static_cast<uint64_t>((x1 - x0) * (y1 - y0)));
            const size_t index = static_cast<size_t>(y) * width + x;
            const int value = gray.pixels[index * 3u];
            if (value >= mean + offset) { bright[index] = 1; ++bright_count; }
            if (value + offset <= mean) { dark[index] = 1; ++dark_count; }
        }
    }
    std::vector<uint8_t> chosen;
    if (!choose_sparse_polarity(bright, dark, bright_count, dark_count, &chosen)) return false;
    *output = binary_from_mask(gray, chosen, "adaptive_3x");
    return true;
}

OcrFilterCandidate color_mask(
    const AIImage& source,
    const OcrColorFilter& filter,
    bool binary,
    bool recover_antialias = false) {
    OcrFilterCandidate output;
    output.width = source.width;
    output.height = source.height;
    output.stride = source.width * 3;
    output.mode = recover_antialias
        ? (binary ? "color_antialias_binary" : "color_antialias_soft")
        : (binary ? "color_binary" : "color_soft");
    output.pixels.resize(
        static_cast<size_t>(output.stride) * static_cast<size_t>(output.height));
    output.foreground_mask.assign(
        static_cast<size_t>(output.width) * output.height, 0);
    for (int32_t y = 0; y < source.height; ++y) {
        for (int32_t x = 0; x < source.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            source_pixel_bgr(source, x, y, &b, &g, &r);
            float best = 0.0f;
            for (const OcrColorRule& rule : filter.rules) {
                const int dr = std::abs(static_cast<int>(r) - rule.r);
                const int dg = std::abs(static_cast<int>(g) - rule.g);
                const int db = std::abs(static_cast<int>(b) - rule.b);
                const int tolerance_r = recover_antialias
                    ? std::min(96, std::max<int>(rule.tolerance_r + 16, rule.tolerance_r * 2))
                    : rule.tolerance_r;
                const int tolerance_g = recover_antialias
                    ? std::min(96, std::max<int>(rule.tolerance_g + 16, rule.tolerance_g * 2))
                    : rule.tolerance_g;
                const int tolerance_b = recover_antialias
                    ? std::min(96, std::max<int>(rule.tolerance_b + 16, rule.tolerance_b * 2))
                    : rule.tolerance_b;
                if (dr > tolerance_r || dg > tolerance_g || db > tolerance_b) continue;
                const float rr = static_cast<float>(dr) / (tolerance_r + 1);
                const float gg = static_cast<float>(dg) / (tolerance_g + 1);
                const float bb = static_cast<float>(db) / (tolerance_b + 1);
                best = std::max(best, 1.0f - std::max(rr, std::max(gg, bb)));
            }
            const uint8_t value = binary
                ? (best > 0.0f ? 255 : 0)
                : static_cast<uint8_t>(std::clamp(best * 255.0f, 0.0f, 255.0f));
            write_gray(&output.pixels, output.width, x, y, value);
            if (best > 0.0f) {
                output.foreground_mask[static_cast<size_t>(y) * output.width + x] = 1;
            }
        }
    }
    summarize_foreground(&output);
    return output;
}

size_t foreground_count(const OcrFilterCandidate& candidate) {
    size_t count = 0;
    for (size_t i = 0; i < candidate.pixels.size(); i += 3u) {
        if (candidate.pixels[i] > 0) ++count;
    }
    return count;
}

size_t utf8_codepoint_count(const char* text) {
    if (text == nullptr) return 0;
    size_t count = 0;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(text);
    while (*p != 0) {
        if ((*p & 0xc0u) != 0x80u && !std::isspace(*p)) ++count;
        ++p;
    }
    return count;
}

bool line_contains_any(
    const std::string& line,
    const std::vector<std::string>& variants) {
    for (const std::string& target : variants) {
        if (!target.empty() && line.find(target) != std::string::npos) return true;
    }
    return false;
}

} // namespace

bool parse_ocr_color_filter(
    const char* text,
    OcrColorFilter* output,
    std::string* error) {
    if (output == nullptr) return false;
    *output = OcrColorFilter{};
    const std::string supplied = text == nullptr ? std::string() : std::string(text);
    if (supplied.size() > kMaxFilterBytes) {
        if (error != nullptr) *error = "OCR color filter is longer than 512 bytes";
        return false;
    }
    const std::string raw = trim_copy(supplied);
    if (raw.empty()) return true;
    size_t begin = 0;
    while (begin <= raw.size()) {
        const size_t end = raw.find('|', begin);
        const std::string item = trim_copy(raw.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.size() != 13 || item[6] != '-') {
            if (error != nullptr) {
                *error = "OCR color filter item " +
                    std::to_string(output->rules.size() + 1) +
                    " is invalid; expected RRGGBB-RRGGBB";
            }
            return false;
        }
        for (size_t i = 0; i < item.size(); ++i) {
            if (i == 6) continue;
            if (!is_hex(item[i])) {
                if (error != nullptr) {
                    *error = "OCR color filter item " +
                        std::to_string(output->rules.size() + 1) +
                        " contains a non-hexadecimal character";
                }
                return false;
            }
        }
        if (output->rules.size() >= kMaxRules) {
            if (error != nullptr) *error = "OCR color filter contains more than 16 colors";
            return false;
        }
        OcrColorRule rule{};
        rule.r = hex_pair(item.data());
        rule.g = hex_pair(item.data() + 2);
        rule.b = hex_pair(item.data() + 4);
        rule.tolerance_r = hex_pair(item.data() + 7);
        rule.tolerance_g = hex_pair(item.data() + 9);
        rule.tolerance_b = hex_pair(item.data() + 11);
        output->rules.push_back(rule);
        if (end == std::string::npos) break;
        begin = end + 1;
        if (begin == raw.size()) {
            if (error != nullptr) *error = "OCR color filter contains an empty item";
            return false;
        }
    }
    output->automatic = false;
    output->canonical = raw;
    return true;
}

std::vector<OcrFilterCandidate> make_ocr_filter_candidates(
    const AIImage& source,
    const OcrColorFilter& filter) {
    std::vector<OcrFilterCandidate> candidates;
    if (source.data == nullptr || source.width <= 0 || source.height <= 0) {
        return candidates;
    }

    const float scale = adaptive_scale(source.width, source.height);
    const OcrFilterCandidate original = copy_bgr(source);
    if (!filter.automatic) {
        OcrFilterCandidate soft = color_mask(source, filter, false);
        const size_t total = static_cast<size_t>(source.width) * source.height;
        const size_t foreground = foreground_count(soft);
        if (foreground > 0 && foreground * 2000 >= total &&
            foreground * 100 < total * 60) {
            candidates.push_back(resize_candidate(
                soft, scale, false, "color_soft_scaled"));
            const OcrFilterCandidate binary = color_mask(source, filter, true);
            candidates.push_back(resize_candidate(
                binary, scale, true, "color_binary_scaled"));
        }
        OcrFilterCandidate antialias_soft = color_mask(source, filter, false, true);
        const size_t recovered_foreground = foreground_count(antialias_soft);
        if (recovered_foreground > 0 && recovered_foreground >= foreground &&
            recovered_foreground * 2000 >= total &&
            recovered_foreground * 100 < total * 60) {
            candidates.push_back(resize_candidate(
                antialias_soft, scale, false, "color_antialias_soft_scaled"));
            const OcrFilterCandidate antialias_binary =
                color_mask(source, filter, true, true);
            candidates.push_back(resize_candidate(
                antialias_binary, scale, true, "color_antialias_binary_scaled"));
        }
    }

    candidates.push_back(original);
    OcrFilterCandidate scaled = resize_candidate(
        original, scale, false, "bgr_scaled");
    if (scaled.width != original.width || scaled.height != original.height) {
        candidates.push_back(scaled);
    }
    OcrFilterCandidate gray = gray_contrast(scaled);
    candidates.push_back(gray);
    for (OcrFilterCandidate& otsu : make_otsu_candidates(gray)) {
        candidates.push_back(std::move(otsu));
    }
    OcrFilterCandidate adaptive;
    if (make_adaptive_candidate(gray, &adaptive)) {
        candidates.push_back(std::move(adaptive));
    }
    return candidates;
}

void remap_ocr_lines_from_candidate(
    std::vector<AIOcrLine>* lines,
    const OcrFilterCandidate& candidate,
    int32_t original_width,
    int32_t original_height) {
    if (lines == nullptr ||
        (candidate.scale_x <= 1.0001f && candidate.scale_y <= 1.0001f)) return;
    for (AIOcrLine& line : *lines) {
        const int32_t left = static_cast<int32_t>(
            std::floor(static_cast<float>(line.box.x) / candidate.scale_x));
        const int32_t top = static_cast<int32_t>(
            std::floor(static_cast<float>(line.box.y) / candidate.scale_y));
        const int32_t right = static_cast<int32_t>(std::ceil(
            static_cast<float>(line.box.x + line.box.w) / candidate.scale_x));
        const int32_t bottom = static_cast<int32_t>(std::ceil(
            static_cast<float>(line.box.y + line.box.h) / candidate.scale_y));
        line.box.x = std::clamp(left, 0, std::max(0, original_width - 1));
        line.box.y = std::clamp(top, 0, std::max(0, original_height - 1));
        line.box.w = std::clamp(
            right - line.box.x, 1, std::max(1, original_width - line.box.x));
        line.box.h = std::clamp(
            bottom - line.box.y, 1, std::max(1, original_height - line.box.y));
    }
}

OcrCandidateEvaluation evaluate_ocr_candidate(
    const std::vector<AIOcrLine>& lines,
    int32_t image_width,
    int32_t image_height,
    float min_confidence,
    const OcrTargetGroups* target_groups,
    const OcrFilterCandidate* source_candidate,
    const OcrRecognitionDiagnostics* diagnostics) {
    OcrCandidateEvaluation result;
    const double image_area = static_cast<double>(std::max(1, image_width)) *
        std::max(1, image_height);
    double confidence_sum = 0.0;
    size_t confidence_chars = 0;
    float maximum_confidence = 0.0f;
    bool full_frame_like = false;
    bool has_output_line = false;
    std::vector<bool> matched(
        target_groups == nullptr ? 0 : target_groups->size(), false);
    const bool measure_foreground_coverage = source_candidate != nullptr &&
        !source_candidate->foreground_mask.empty() &&
        source_candidate->active_columns > 0;
    std::vector<uint8_t> covered_columns;
    if (measure_foreground_coverage) {
        covered_columns.assign(
            static_cast<size_t>(std::max(0, image_width)), 0);
    }

    for (const AIOcrLine& line : lines) {
        if (line.text[0] == '\0' || line.box.w <= 0 || line.box.h <= 0) continue;
        const float confidence = std::clamp(
            std::isfinite(line.confidence) ? line.confidence : 0.0f, 0.0f, 1.0f);
        maximum_confidence = std::max(maximum_confidence, confidence);
        const size_t characters = utf8_codepoint_count(line.text);
        if (characters > 0 && confidence >= min_confidence) has_output_line = true;
        result.unicode_chars += characters;
        const double area_ratio =
            static_cast<double>(line.box.w) * line.box.h / image_area;
        if (area_ratio >= 0.85 ||
            (line.box.w >= image_width * 9 / 10 &&
             line.box.h >= image_height * 9 / 10)) {
            full_frame_like = true;
        }
        if (confidence >= 0.20f && characters > 0) {
            result.quality += characters * confidence * confidence;
            confidence_sum += characters * confidence;
            confidence_chars += characters;
        }
        if (confidence >= 0.50f && characters > 0 && area_ratio < 0.85) {
            result.high_confidence_chars += characters;
            ++result.plausible_lines;
            result.quality += 1.5;
        }
        if (target_groups != nullptr && confidence >= min_confidence) {
            const std::string text(line.text);
            for (size_t i = 0; i < target_groups->size(); ++i) {
                if (!matched[i] && line_contains_any(text, (*target_groups)[i])) {
                    matched[i] = true;
                }
            }
        }
        if (measure_foreground_coverage) {
            const int32_t left = std::clamp(line.box.x, 0, image_width);
            const int32_t right = std::clamp(
                line.box.x + line.box.w, 0, image_width);
            for (int32_t x = left; x < right; ++x) {
                covered_columns[static_cast<size_t>(x)] = 1;
            }
        }
    }
    for (const bool value : matched) {
        if (value) ++result.target_groups_matched;
    }
    result.mean_confidence = confidence_chars == 0
        ? 0.0f
        : static_cast<float>(confidence_sum / confidence_chars);

    if (measure_foreground_coverage) {
        size_t active_columns = 0;
        size_t covered_active_columns = 0;
        size_t leading = 0;
        size_t trailing = 0;
        bool saw_covered = false;
        size_t pending_gap = 0;
        for (int32_t original_x = 0; original_x < image_width; ++original_x) {
            const int32_t candidate_x = std::clamp(
                static_cast<int32_t>(std::floor(
                    original_x * source_candidate->scale_x)),
                0, source_candidate->width - 1);
            bool active = false;
            for (int32_t y = 0; y < source_candidate->height; ++y) {
                if (source_candidate->foreground_mask[
                        static_cast<size_t>(y) * source_candidate->width +
                        candidate_x] != 0) {
                    active = true;
                    break;
                }
            }
            if (!active) continue;
            ++active_columns;
            if (covered_columns[static_cast<size_t>(original_x)] != 0) {
                ++covered_active_columns;
                saw_covered = true;
                pending_gap = 0;
            } else if (!saw_covered) {
                ++leading;
            } else {
                ++pending_gap;
            }
        }
        trailing = pending_gap;
        result.uncovered_leading_columns = leading;
        result.uncovered_trailing_columns = trailing;
        result.foreground_column_coverage = active_columns == 0
            ? 1.0f
            : static_cast<float>(covered_active_columns) / active_columns;
        const size_t edge_gap_limit = std::max<size_t>(
            2, static_cast<size_t>(std::ceil(image_height * 0.15f)));
        result.spatial_complete = result.foreground_column_coverage >= 0.92f &&
            leading <= edge_gap_limit && trailing <= edge_gap_limit;
    }

    // Detection geometry is measured against the original source support as
    // well. A strict color candidate may erase antialias pixels, so using only
    // its own mask would incorrectly certify an incomplete tail as complete.
    if (source_candidate != nullptr && source_candidate->foreground_mask.empty()) {
        result.spatial_complete = true;
    }
    if (lines.empty() || result.unicode_chars == 0) {
        result.foreground_column_coverage = 0.0f;
        result.spatial_complete = false;
    }

    if (lines.size() >= 2) {
        for (size_t i = 1; i < lines.size(); ++i) {
            const AIOcrLine& previous = lines[i - 1];
            const AIOcrLine& next = lines[i];
            const int overlap_y = std::max(
                0, std::min(previous.box.y + previous.box.h, next.box.y + next.box.h) -
                   std::max(previous.box.y, next.box.y));
            if (overlap_y * 2 < std::min(previous.box.h, next.box.h)) continue;
            const int previous_padding = std::max(
                1, static_cast<int>(std::ceil(previous.box.h * 0.25f)));
            const int next_padding = std::max(
                1, static_cast<int>(std::ceil(next.box.h * 0.25f)));
            const int gap = next.box.x - (previous.box.x + previous.box.w);
            if (gap < previous_padding + next_padding) ++result.fragmented_pairs;
        }
        result.padded_overlap_ratio = static_cast<float>(result.fragmented_pairs) /
            static_cast<float>(lines.size() - 1);
    }

    bool single_line_risk = false;
    bool single_line_edge_gap_risk = false;
    if (lines.size() == 1 && std::max(image_width, image_height) <= 512) {
        const AIOcrLine& line = lines.front();
        const int32_t unused_vertical = std::max(0, image_height - line.box.h);
        single_line_risk = maximum_confidence < 0.80f &&
            unused_vertical >= line.box.h + 8;
        const int32_t left_margin = std::max(0, line.box.x);
        const int32_t right_margin = std::max(
            0, image_width - (line.box.x + line.box.w));
        const int32_t significant_tail = std::max(
            static_cast<int32_t>(std::ceil(line.box.h * 1.5f)),
            static_cast<int32_t>(std::ceil(image_width * 0.15f)));
        single_line_edge_gap_risk = image_width >= image_height * 3 &&
            left_margin <= std::max(2, line.box.h / 2) &&
            right_margin >= significant_tail &&
            line.box.w * 100 < image_width * 85;
    }
    const bool detector_fragmented = diagnostics != nullptr &&
        diagnostics->detected_box_count >= 3 && result.padded_overlap_ratio >= 0.50f;
    result.suspicious = lines.empty() || result.unicode_chars == 0 || !has_output_line ||
        maximum_confidence < 0.35f || full_frame_like || single_line_risk ||
        single_line_edge_gap_risk ||
        !result.spatial_complete || detector_fragmented;
    result.strong = !result.suspicious && result.high_confidence_chars >= 4 &&
        result.mean_confidence >= 0.75f &&
        (target_groups == nullptr || target_groups->empty() ||
         result.target_groups_matched == target_groups->size());
    return result;
}

bool is_better_ocr_candidate(
    const OcrCandidateEvaluation& candidate,
    const OcrCandidateEvaluation& current_best) {
    if (candidate.target_groups_matched != current_best.target_groups_matched) {
        return candidate.target_groups_matched > current_best.target_groups_matched;
    }
    if (candidate.spatial_complete != current_best.spatial_complete) {
        return candidate.spatial_complete;
    }
    if (candidate.foreground_column_coverage >=
            current_best.foreground_column_coverage + 0.05f) return true;
    if (current_best.foreground_column_coverage >=
            candidate.foreground_column_coverage + 0.05f) return false;
    if (candidate.padded_overlap_ratio + 0.20f <
            current_best.padded_overlap_ratio) return true;
    if (current_best.padded_overlap_ratio + 0.20f <
            candidate.padded_overlap_ratio) return false;
    if (candidate.suspicious != current_best.suspicious) return !candidate.suspicious;
    if (candidate.mean_confidence >= current_best.mean_confidence + 0.05f &&
        candidate.unicode_chars >= current_best.unicode_chars) return true;
    if (candidate.unicode_chars > current_best.unicode_chars &&
        candidate.mean_confidence + 0.10f >= current_best.mean_confidence) return true;
    if (current_best.quality <= 0.0) return candidate.quality > 0.0;
    return candidate.quality >= current_best.quality * 1.10;
}

std::string explain_candidate_rejection(
    const OcrCandidateEvaluation& rejected,
    const OcrCandidateEvaluation& preferred) {
    if (rejected.target_groups_matched < preferred.target_groups_matched) {
        return "fewer target groups matched";
    }
    if (!rejected.spatial_complete && preferred.spatial_complete) {
        return "spatially incomplete";
    }
    if (rejected.foreground_column_coverage + 0.05f <
        preferred.foreground_column_coverage) {
        return "lower foreground-column coverage";
    }
    if (rejected.padded_overlap_ratio >
        preferred.padded_overlap_ratio + 0.20f) {
        return "higher fragment-overlap penalty";
    }
    if (rejected.suspicious && !preferred.suspicious) {
        return "suspicious geometry or confidence";
    }
    if (rejected.mean_confidence + 0.05f < preferred.mean_confidence) {
        return "lower mean confidence";
    }
    if (rejected.unicode_chars < preferred.unicode_chars) {
        return "fewer recognized characters";
    }
    return "not preferred by generic candidate tie-break";
}

void filter_ocr_lines_by_confidence(
    std::vector<AIOcrLine>* lines,
    float min_confidence) {
    if (lines == nullptr) return;
    lines->erase(
        std::remove_if(
            lines->begin(),
            lines->end(),
            [min_confidence](const AIOcrLine& line) {
                return !std::isfinite(line.confidence) ||
                    line.confidence < min_confidence;
            }),
        lines->end());
}

int32_t run_ocr_candidate_pipeline(
    const AIImage& source,
    const OcrColorFilter& filter,
    float min_confidence,
    const OcrTargetGroups* target_groups,
    const OcrRecognizeCallback& recognize,
    std::vector<AIOcrLine>* output,
    std::string* error,
    OcrPipelineSelection* selection) {
    if (output == nullptr || !recognize || min_confidence < 0.0f ||
        min_confidence > 1.0f) return AI_ERR_INVALID_ARGUMENT;
    output->clear();
    if (selection != nullptr) *selection = OcrPipelineSelection{};
    bool have_success = false;
    bool have_best = false;
    OcrCandidateEvaluation best_evaluation;
    OcrRecognitionDiagnostics best_diagnostics;
    std::string best_candidate_type;
    std::vector<AIOcrLine> best_lines;
    std::string failures;
    int32_t last_failure = AI_ERR_RUNTIME;
    OcrFilterCandidate source_support;
    bool source_support_initialized = false;
    size_t executed_candidates = 0;
    size_t best_trace_index = std::numeric_limits<size_t>::max();

    // Build support lazily so the established automatic fast path does not
    // pay preprocessing cost. Explicit filters and suspicious automatic
    // results always obtain independent source coverage before selection.
    const auto ensure_source_support = [&]() {
        if (source_support_initialized) return;
        source_support_initialized = true;
        const OcrFilterCandidate original = copy_bgr(source);
        OcrFilterCandidate native_gray = gray_plain(original);
        native_gray.mode = "source_gray";
        auto support_candidates = make_otsu_candidates(native_gray);
        if (!support_candidates.empty()) {
            OcrFilterCandidate support = std::move(support_candidates.front());
            support.scale_x = support.scale_y = 1.0f;
            source_support = std::move(support);
        }
    };

    const auto process_candidate = [&](const OcrFilterCandidate& candidate) {
        std::vector<AIOcrLine> attempt;
        OcrRecognitionDiagnostics diagnostics;
        std::string attempt_error;
        const int32_t status = recognize(
            candidate.view(), &attempt, &diagnostics, &attempt_error);
        if (status < 0) {
            last_failure = status;
            if (!failures.empty()) failures += "; ";
            failures += candidate.mode;
            failures += ": ";
            failures += attempt_error.empty() ? "recognition failed" : attempt_error;
            if (selection != nullptr) {
                OcrCandidateTrace trace;
                trace.candidate_type = candidate.mode;
                trace.status = status;
                trace.rejection_reason = attempt_error.empty()
                    ? "recognition failed"
                    : attempt_error;
                selection->attempts.push_back(std::move(trace));
            }
            return OcrCandidateEvaluation{};
        }
        have_success = true;
        remap_ocr_lines_from_candidate(
            &attempt, candidate, source.width, source.height);
        const auto evaluate_attempt = [&](const std::vector<AIOcrLine>& lines,
                                          const OcrRecognitionDiagnostics& attempt_diagnostics) {
            OcrCandidateEvaluation scored = evaluate_ocr_candidate(
                lines,
                source.width,
                source.height,
                min_confidence,
                target_groups,
                &candidate,
                &attempt_diagnostics);
            if (!source_support.foreground_mask.empty()) {
                const OcrCandidateEvaluation source_evaluation = evaluate_ocr_candidate(
                    lines,
                    source.width,
                    source.height,
                    min_confidence,
                    target_groups,
                    &source_support,
                    &attempt_diagnostics);
                scored.foreground_column_coverage = std::min(
                    scored.foreground_column_coverage,
                    source_evaluation.foreground_column_coverage);
                scored.uncovered_leading_columns = std::max(
                    scored.uncovered_leading_columns,
                    source_evaluation.uncovered_leading_columns);
                scored.uncovered_trailing_columns = std::max(
                    scored.uncovered_trailing_columns,
                    source_evaluation.uncovered_trailing_columns);
                scored.spatial_complete = scored.spatial_complete &&
                    source_evaluation.spatial_complete;
                scored.suspicious = scored.suspicious ||
                    !scored.spatial_complete;
                scored.strong = scored.strong && scored.spatial_complete;
            }
            return scored;
        };
        OcrCandidateEvaluation evaluation = evaluate_attempt(attempt, diagnostics);
        const auto append_trace = [&](const std::string& type,
                                      const OcrCandidateEvaluation& scored,
                                      const OcrRecognitionDiagnostics& trace_diagnostics) {
            if (selection == nullptr) return std::numeric_limits<size_t>::max();
            OcrCandidateTrace trace;
            trace.candidate_type = type;
            trace.status = AI_OK;
            trace.foreground_column_coverage = scored.foreground_column_coverage;
            trace.padded_overlap_ratio = scored.padded_overlap_ratio;
            trace.mean_confidence = scored.mean_confidence;
            trace.fragmented_pairs = scored.fragmented_pairs;
            trace.unicode_chars = scored.unicode_chars;
            trace.spatial_complete = scored.spatial_complete;
            trace.used_eight_connectivity =
                trace_diagnostics.used_eight_connectivity;
            trace.used_merged_line_recognition =
                trace_diagnostics.used_merged_line_recognition;
            trace.detection_us = trace_diagnostics.detection_us;
            trace.recognition_us = trace_diagnostics.recognition_us;
            trace.postprocess_us = trace_diagnostics.postprocess_us;
            selection->attempts.push_back(std::move(trace));
            return selection->attempts.size() - 1;
        };
        size_t chosen_trace_index = std::numeric_limits<size_t>::max();
        if (!diagnostics.split_box_lines.empty()) {
            const OcrCandidateEvaluation merged_evaluation = evaluation;
            const OcrRecognitionDiagnostics merged_diagnostics = diagnostics;
            remap_ocr_lines_from_candidate(
                &diagnostics.split_box_lines,
                candidate,
                source.width,
                source.height);
            OcrRecognitionDiagnostics split_diagnostics = diagnostics;
            split_diagnostics.used_merged_line_recognition = false;
            split_diagnostics.split_box_lines.clear();
            const OcrCandidateEvaluation split_evaluation =
                evaluate_attempt(diagnostics.split_box_lines, split_diagnostics);
            const size_t merged_trace_index = append_trace(
                std::string(candidate.mode) + "/merged",
                merged_evaluation,
                merged_diagnostics);
            const size_t split_trace_index = append_trace(
                std::string(candidate.mode) + "/split",
                split_evaluation,
                split_diagnostics);
            if (is_better_ocr_candidate(split_evaluation, merged_evaluation)) {
                if (selection != nullptr) {
                    selection->attempts[merged_trace_index].rejection_reason =
                        explain_candidate_rejection(
                            merged_evaluation, split_evaluation);
                }
                attempt = std::move(diagnostics.split_box_lines);
                diagnostics = std::move(split_diagnostics);
                evaluation = split_evaluation;
                chosen_trace_index = split_trace_index;
            } else {
                if (selection != nullptr) {
                    selection->attempts[split_trace_index].rejection_reason =
                        explain_candidate_rejection(
                            split_evaluation, merged_evaluation);
                }
                chosen_trace_index = merged_trace_index;
            }
        } else {
            chosen_trace_index = append_trace(candidate.mode, evaluation, diagnostics);
        }
        if (!have_best || is_better_ocr_candidate(evaluation, best_evaluation)) {
            if (selection != nullptr && have_best &&
                best_trace_index < selection->attempts.size()) {
                selection->attempts[best_trace_index].rejection_reason =
                    explain_candidate_rejection(best_evaluation, evaluation);
            }
            have_best = true;
            best_evaluation = evaluation;
            if (selection != nullptr) {
                best_diagnostics = diagnostics;
                best_candidate_type = candidate.mode;
                best_trace_index = chosen_trace_index;
            }
            best_lines = attempt;
        } else if (selection != nullptr &&
                   chosen_trace_index < selection->attempts.size()) {
            selection->attempts[chosen_trace_index].rejection_reason =
                explain_candidate_rejection(evaluation, best_evaluation);
        }
        return evaluation;
    };

    const auto publish_selection = [&]() {
        if (selection == nullptr || !have_best) return;
        selection->candidate_type = best_candidate_type;
        selection->foreground_column_coverage =
            best_evaluation.foreground_column_coverage;
        selection->padded_overlap_ratio = best_evaluation.padded_overlap_ratio;
        selection->spatial_complete = best_evaluation.spatial_complete;
        selection->used_eight_connectivity =
            best_diagnostics.used_eight_connectivity;
        selection->used_merged_line_recognition =
            best_diagnostics.used_merged_line_recognition;
        if (best_trace_index < selection->attempts.size()) {
            selection->attempts[best_trace_index].selected = true;
            selection->attempts[best_trace_index].rejection_reason.clear();
        }
    };

    // 自动模式的绝大多数正常图片只复制并识别原图。只有结果可疑时才分配
    // 三倍图、灰度图和二值图，避免为了不会执行的候选支付预处理成本。
    bool automatic_original_processed = false;
    if (filter.automatic) {
        const OcrFilterCandidate original = copy_bgr(source);
        const OcrCandidateEvaluation original_evaluation = process_candidate(original);
        ++executed_candidates;
        automatic_original_processed = true;
        if (have_success && !original_evaluation.suspicious) {
            filter_ocr_lines_by_confidence(&best_lines, min_confidence);
            *output = std::move(best_lines);
            publish_selection();
            return static_cast<int32_t>(output->size());
        }
        ensure_source_support();
        // Small high-contrast UI labels are frequently bright text blended
        // into a dark background. Reuse the same generic color-mask path with
        // a neutral bright target; this is a preprocessing hypothesis, not a
        // business text or image-specific correction.
        if (source.width <= 512 && source.height <= 256) {
            OcrColorFilter bright_filter;
            bright_filter.automatic = false;
            OcrColorRule bright_rule;
            bright_rule.r = bright_rule.g = bright_rule.b = 255;
            bright_rule.tolerance_r = bright_rule.tolerance_g =
                bright_rule.tolerance_b = 128;
            bright_filter.rules.push_back(bright_rule);
            OcrFilterCandidate bright = color_mask(
                source, bright_filter, false, false);
            {
                const float scale = 1.0f;
                bright = resize_candidate(
                    bright, scale, false, "auto_bright_soft_scaled");
                const OcrCandidateEvaluation bright_evaluation =
                    process_candidate(bright);
                ++executed_candidates;
                if (have_success && bright_evaluation.strong) {
                    filter_ocr_lines_by_confidence(&best_lines, min_confidence);
                    *output = std::move(best_lines);
                    publish_selection();
                    return static_cast<int32_t>(output->size());
                }
            }
        }
    } else {
        ensure_source_support();
    }

    const std::vector<OcrFilterCandidate> candidates =
        make_ocr_filter_candidates(source, filter);
    if (candidates.empty()) {
        if (error != nullptr) *error = "OCR could not create a preprocessing candidate";
        return AI_ERR_IMAGE_FORMAT;
    }
    for (const OcrFilterCandidate& candidate : candidates) {
        if (automatic_original_processed &&
            std::strcmp(candidate.mode, "original") == 0) continue;
        if (executed_candidates >= kMaxExecutedCandidates) break;
        const OcrCandidateEvaluation evaluation = process_candidate(candidate);
        ++executed_candidates;
        if (have_success && evaluation.strong) break;
    }

    if (!have_success) {
        if (error != nullptr) {
            *error = failures.empty()
                ? "OCR recognition failed for all preprocessing candidates"
                : "OCR preprocessing candidates failed: " + failures;
        }
        return last_failure;
    }
    if (!have_best) return 0;
    filter_ocr_lines_by_confidence(&best_lines, min_confidence);
    *output = std::move(best_lines);
    publish_selection();
    return static_cast<int32_t>(output->size());
}

} // namespace ai
