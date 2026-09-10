#include "cv_fast.h"

#include "image_view.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace ai {

namespace {

// 统一校验图像，并归一化/裁剪可选 ROI。
int32_t validate_image_and_roi(const AIImage& image, const AIRect* roi, AIRect* normalized) {
    if (!validate_image(image)) {
        return AI_ERR_IMAGE_FORMAT;
    }

    *normalized = normalize_roi(image, roi);
    if (normalized->w <= 0 || normalized->h <= 0) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    return AI_OK;
}

// 模板匹配分数统一归一化到 [0,1]。
bool valid_score(float min_score) {
    return min_score >= 0.0f && min_score <= 1.0f && std::isfinite(min_score);
}

// 一个已排序的骨架连通分量，以及它在原图坐标中的边界。
struct TracePath {
    std::vector<int32_t> pixels;
    AIRect bounds{};
};

// 将 ROI 内局部坐标转换为平面缓冲区索引。
int32_t trace_index(int32_t x, int32_t y, int32_t width) {
    return y * width + x;
}

// 从轨迹平面缓冲区索引提取 x 坐标。
int32_t trace_x(int32_t index, int32_t width) {
    return index % width;
}

// 从轨迹平面缓冲区索引提取 y 坐标。
int32_t trace_y(int32_t index, int32_t width) {
    return index / width;
}

// 稳定的从上到下、从左到右排序，用于生成确定性 JSON。
bool trace_pixel_less(int32_t a, int32_t b, int32_t width) {
    const int32_t ay = trace_y(a, width);
    const int32_t by = trace_y(b, width);
    if (ay != by) {
        return ay < by;
    }
    return trace_x(a, width) < trace_x(b, width);
}

// 轨迹提取收到 threshold=-1 时使用的 Otsu 自动阈值选择。
int32_t otsu_threshold(const std::array<uint32_t, 256>& histogram, int64_t total) {
    if (total <= 0) {
        return 127;
    }

    double sum = 0.0;
    for (int32_t i = 0; i < 256; ++i) {
        sum += static_cast<double>(i) * static_cast<double>(histogram[static_cast<size_t>(i)]);
    }

    double sum_b = 0.0;
    int64_t weight_b = 0;
    double best_variance = -1.0;
    int32_t best_threshold = 127;
    for (int32_t t = 0; t < 256; ++t) {
        weight_b += static_cast<int64_t>(histogram[static_cast<size_t>(t)]);
        if (weight_b == 0) {
            continue;
        }

        const int64_t weight_f = total - weight_b;
        if (weight_f == 0) {
            break;
        }

        sum_b += static_cast<double>(t) * static_cast<double>(histogram[static_cast<size_t>(t)]);
        const double mean_b = sum_b / static_cast<double>(weight_b);
        const double mean_f = (sum - sum_b) / static_cast<double>(weight_f);
        const double diff = mean_b - mean_f;
        const double variance = static_cast<double>(weight_b) * static_cast<double>(weight_f) * diff * diff;
        if (variance > best_variance) {
            best_variance = variance;
            best_threshold = t;
        }
    }
    return best_threshold;
}

// 将一个源像素读取为灰度值，不依赖原始图像格式。
uint8_t read_gray_pixel(const AIImage& image, int32_t x, int32_t y, int channels) {
    const uint8_t* src = image_row_ptr(image, y) + x * channels;
    uint8_t b = 0;
    uint8_t g = 0;
    uint8_t r = 0;
    read_bgr(src, image.format, &b, &g, &r);
    return bgr_to_gray(b, g, r);
}

// 为笔画/轨迹提取构建二值前景掩码。
//
// invert=true 表示浅色背景上的深色前景；invert=false 表示深色背景上的浅色前景。
int32_t build_trace_foreground(
    const AIImage& image,
    const AIRect& roi,
    int32_t threshold,
    bool invert,
    std::vector<uint8_t>* foreground,
    int32_t* actual_threshold,
    int32_t* foreground_count) {
    if (foreground == nullptr || actual_threshold == nullptr || foreground_count == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (threshold < -1 || threshold > 255) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    const int32_t width = roi.w;
    const int32_t height = roi.h;
    const int64_t area64 = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (area64 <= 0 || area64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    const int32_t area = static_cast<int32_t>(area64);
    std::vector<uint8_t> gray(static_cast<size_t>(area));
    std::array<uint32_t, 256> histogram{};
    const int channels = channels_for_format(image.format);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const uint8_t value = read_gray_pixel(image, roi.x + x, roi.y + y, channels);
            gray[static_cast<size_t>(trace_index(x, y, width))] = value;
            ++histogram[static_cast<size_t>(value)];
        }
    }

    const int32_t chosen_threshold = threshold >= 0 ? threshold : otsu_threshold(histogram, area64);
    foreground->assign(static_cast<size_t>(area), 0);
    int32_t count = 0;
    for (int32_t i = 0; i < area; ++i) {
        const uint8_t value = gray[static_cast<size_t>(i)];
        const bool is_foreground = invert ? value <= chosen_threshold : value >= chosen_threshold;
        if (is_foreground) {
            (*foreground)[static_cast<size_t>(i)] = 1;
            ++count;
        }
    }

    *actual_threshold = chosen_threshold;
    *foreground_count = count;
    return AI_OK;
}

// Zhang-Suen 细化算法的一次子迭代。两个交替子迭代会在保持连通性的同时移除边界像素。
bool thin_trace_subiteration(std::vector<uint8_t>* image, int32_t width, int32_t height, bool second_step, std::vector<uint8_t>* marker) {
    if (image == nullptr || marker == nullptr) {
        return false;
    }

    std::fill(marker->begin(), marker->end(), uint8_t{0});
    bool changed = false;
    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            const int32_t i = trace_index(x, y, width);
            if ((*image)[static_cast<size_t>(i)] == 0) {
                continue;
            }

            const uint8_t p2 = (*image)[static_cast<size_t>(trace_index(x, y - 1, width))];
            const uint8_t p3 = (*image)[static_cast<size_t>(trace_index(x + 1, y - 1, width))];
            const uint8_t p4 = (*image)[static_cast<size_t>(trace_index(x + 1, y, width))];
            const uint8_t p5 = (*image)[static_cast<size_t>(trace_index(x + 1, y + 1, width))];
            const uint8_t p6 = (*image)[static_cast<size_t>(trace_index(x, y + 1, width))];
            const uint8_t p7 = (*image)[static_cast<size_t>(trace_index(x - 1, y + 1, width))];
            const uint8_t p8 = (*image)[static_cast<size_t>(trace_index(x - 1, y, width))];
            const uint8_t p9 = (*image)[static_cast<size_t>(trace_index(x - 1, y - 1, width))];
            const int32_t neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            if (neighbor_count < 2 || neighbor_count > 6) {
                continue;
            }

            const uint8_t neighbors[9] = {p2, p3, p4, p5, p6, p7, p8, p9, p2};
            int32_t transitions = 0;
            for (int32_t n = 0; n < 8; ++n) {
                if (neighbors[n] == 0 && neighbors[n + 1] != 0) {
                    ++transitions;
                }
            }
            if (transitions != 1) {
                continue;
            }

            if (!second_step) {
                if (p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0) {
                    continue;
                }
            } else {
                if (p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0) {
                    continue;
                }
            }

            (*marker)[static_cast<size_t>(i)] = 1;
            changed = true;
        }
    }

    if (changed) {
        const size_t count = image->size();
        for (size_t i = 0; i < count; ++i) {
            if ((*marker)[i] != 0) {
                (*image)[i] = 0;
            }
        }
    }
    return changed;
}

// 反复细化二值前景掩码，得到近似 1 像素宽的骨架。
void thin_trace_foreground(std::vector<uint8_t>* image, int32_t width, int32_t height) {
    if (image == nullptr || width < 3 || height < 3) {
        return;
    }

    std::vector<uint8_t> marker(image->size(), 0);
    const int32_t max_iterations = std::min<int32_t>(128, std::max(width, height));
    for (int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        const bool changed_a = thin_trace_subiteration(image, width, height, false, &marker);
        const bool changed_b = thin_trace_subiteration(image, width, height, true, &marker);
        if (!changed_a && !changed_b) {
            break;
        }
    }
}

// 统计属于当前连通分量的 8 邻域像素数。
int32_t count_marked_neighbors(const std::vector<int32_t>& mark, int32_t stamp, int32_t index, int32_t width, int32_t height) {
    const int32_t x = trace_x(index, width);
    const int32_t y = trace_y(index, width);
    int32_t count = 0;
    for (int32_t dy = -1; dy <= 1; ++dy) {
        for (int32_t dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int32_t nx = x + dx;
            const int32_t ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                continue;
            }
            if (mark[static_cast<size_t>(trace_index(nx, ny, width))] == stamp) {
                ++count;
            }
        }
    }
    return count;
}

// 返回已排序的未访问邻居，优先选择端点，再选择分叉点。
std::vector<int32_t> sorted_unvisited_neighbors(
    const std::vector<int32_t>& mark,
    const std::vector<int32_t>& used,
    int32_t stamp,
    int32_t used_stamp,
    int32_t index,
    int32_t width,
    int32_t height) {
    const int32_t x = trace_x(index, width);
    const int32_t y = trace_y(index, width);
    std::vector<int32_t> neighbors;
    neighbors.reserve(8);
    for (int32_t dy = -1; dy <= 1; ++dy) {
        for (int32_t dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int32_t nx = x + dx;
            const int32_t ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                continue;
            }
            const int32_t next = trace_index(nx, ny, width);
            if (mark[static_cast<size_t>(next)] == stamp && used[static_cast<size_t>(next)] != used_stamp) {
                neighbors.push_back(next);
            }
        }
    }
    std::sort(neighbors.begin(), neighbors.end(), [&](int32_t a, int32_t b) {
        const int32_t da = count_marked_neighbors(mark, stamp, a, width, height);
        const int32_t db = count_marked_neighbors(mark, stamp, b, width, height);
        if (da != db) {
            return da < db;
        }
        return trace_pixel_less(a, b, width);
    });
    return neighbors;
}

// 选择稳定的路径起点：优先端点，否则选择最靠左上的像素。
int32_t choose_trace_start(const std::vector<int32_t>& component, const std::vector<int32_t>& mark, int32_t stamp, int32_t width, int32_t height) {
    int32_t fallback = component.empty() ? 0 : component[0];
    int32_t endpoint = -1;
    for (int32_t pixel : component) {
        if (trace_pixel_less(pixel, fallback, width)) {
            fallback = pixel;
        }
        const int32_t degree = count_marked_neighbors(mark, stamp, pixel, width, height);
        if (degree <= 1 && (endpoint < 0 || trace_pixel_less(pixel, endpoint, width))) {
            endpoint = pixel;
        }
    }
    return endpoint >= 0 ? endpoint : fallback;
}

// 使用确定性 DFS 遍历排序单个骨架连通分量。
std::vector<int32_t> order_trace_component(
    const std::vector<int32_t>& component,
    const std::vector<int32_t>& mark,
    int32_t stamp,
    int32_t width,
    int32_t height,
    std::vector<int32_t>* used,
    int32_t used_stamp) {
    std::vector<int32_t> ordered;
    ordered.reserve(component.size());
    if (component.empty() || used == nullptr) {
        return ordered;
    }

    std::vector<int32_t> stack;
    stack.reserve(component.size());
    stack.push_back(choose_trace_start(component, mark, stamp, width, height));
    while (!stack.empty()) {
        const int32_t pixel = stack.back();
        stack.pop_back();
        if ((*used)[static_cast<size_t>(pixel)] == used_stamp) {
            continue;
        }

        (*used)[static_cast<size_t>(pixel)] = used_stamp;
        ordered.push_back(pixel);
        std::vector<int32_t> neighbors = sorted_unvisited_neighbors(mark, *used, stamp, used_stamp, pixel, width, height);
        for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
            stack.push_back(*it);
        }
    }

    return ordered;
}

// 查找骨架连通分量，排序每个分量，并将 ROI 局部坐标转换为原图边界。
std::vector<TracePath> extract_trace_paths(const std::vector<uint8_t>& skeleton, int32_t width, int32_t height, const AIRect& roi) {
    const int32_t area = width * height;
    std::vector<uint8_t> visited(static_cast<size_t>(area), 0);
    std::vector<int32_t> component_mark(static_cast<size_t>(area), 0);
    std::vector<int32_t> path_used(static_cast<size_t>(area), 0);
    std::vector<TracePath> paths;
    int32_t mark_stamp = 0;
    int32_t used_stamp = 0;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const int32_t seed = trace_index(x, y, width);
            if (skeleton[static_cast<size_t>(seed)] == 0 || visited[static_cast<size_t>(seed)] != 0) {
                continue;
            }

            ++mark_stamp;
            std::vector<int32_t> component;
            std::vector<int32_t> stack;
            stack.push_back(seed);
            visited[static_cast<size_t>(seed)] = 1;
            int32_t min_x = x;
            int32_t min_y = y;
            int32_t max_x = x;
            int32_t max_y = y;

            while (!stack.empty()) {
                const int32_t pixel = stack.back();
                stack.pop_back();
                component.push_back(pixel);
                component_mark[static_cast<size_t>(pixel)] = mark_stamp;
                const int32_t px = trace_x(pixel, width);
                const int32_t py = trace_y(pixel, width);
                min_x = std::min(min_x, px);
                min_y = std::min(min_y, py);
                max_x = std::max(max_x, px);
                max_y = std::max(max_y, py);

                for (int32_t dy = -1; dy <= 1; ++dy) {
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int32_t nx = px + dx;
                        const int32_t ny = py + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                            continue;
                        }
                        const int32_t next = trace_index(nx, ny, width);
                        if (skeleton[static_cast<size_t>(next)] != 0 && visited[static_cast<size_t>(next)] == 0) {
                            visited[static_cast<size_t>(next)] = 1;
                            stack.push_back(next);
                        }
                    }
                }
            }

            if (component.empty()) {
                continue;
            }

            ++used_stamp;
            TracePath path{};
            path.pixels = order_trace_component(component, component_mark, mark_stamp, width, height, &path_used, used_stamp);
            path.bounds = AIRect{roi.x + min_x, roi.y + min_y, max_x - min_x + 1, max_y - min_y + 1};
            if (!path.pixels.empty()) {
                paths.push_back(std::move(path));
            }
        }
    }

    std::sort(paths.begin(), paths.end(), [](const TracePath& a, const TracePath& b) {
        if (a.bounds.y != b.bounds.y) {
            return a.bounds.y < b.bounds.y;
        }
        if (a.bounds.x != b.bounds.x) {
            return a.bounds.x < b.bounds.x;
        }
        return a.pixels.size() > b.pixels.size();
    });
    return paths;
}

// 统计最终骨架掩码中的非零像素。
int32_t skeleton_pixel_count(const std::vector<uint8_t>& skeleton) {
    int32_t count = 0;
    for (uint8_t value : skeleton) {
        if (value != 0) {
            ++count;
        }
    }
    return count;
}

// 将轨迹路径序列化为紧凑 JSON，供 Python/易语言使用。
// max_points 会通过均匀采样长路径来限制输出大小。
int32_t append_trace_json(
    const std::vector<TracePath>& paths,
    const AIRect& roi,
    int32_t actual_threshold,
    bool invert,
    int32_t max_points,
    int32_t foreground_count,
    int32_t skeleton_count,
    std::string* output) {
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    const int32_t point_limit = max_points <= 0 ? 4096 : std::min<int32_t>(max_points, 100000);
    std::ostringstream oss;
    oss << "{\"width\":" << roi.w
        << ",\"height\":" << roi.h
        << ",\"roi\":{\"x\":" << roi.x << ",\"y\":" << roi.y << ",\"w\":" << roi.w << ",\"h\":" << roi.h << "}"
        << ",\"threshold\":" << actual_threshold
        << ",\"invert\":" << (invert ? "true" : "false")
        << ",\"foreground_pixels\":" << foreground_count
        << ",\"skeleton_pixels\":" << skeleton_count
        << ",\"max_points\":" << point_limit
        << ",\"paths\":[";

    int32_t emitted_points = 0;
    int32_t emitted_paths = 0;
    bool first_path = true;
    for (const TracePath& path : paths) {
        if (emitted_points >= point_limit) {
            break;
        }

        const int32_t path_size = static_cast<int32_t>(path.pixels.size());
        if (path_size <= 0) {
            continue;
        }
        const int32_t allowed = std::min<int32_t>(path_size, point_limit - emitted_points);
        if (allowed <= 0) {
            break;
        }

        if (!first_path) {
            oss << ',';
        }
        first_path = false;
        ++emitted_paths;
        oss << "{\"bounds\":{\"x\":" << path.bounds.x
            << ",\"y\":" << path.bounds.y
            << ",\"w\":" << path.bounds.w
            << ",\"h\":" << path.bounds.h
            << "},\"points\":[";

        for (int32_t i = 0; i < allowed; ++i) {
            int32_t path_index = i;
            if (allowed < path_size && allowed > 1) {
                path_index = static_cast<int32_t>((static_cast<int64_t>(i) * static_cast<int64_t>(path_size - 1)) / static_cast<int64_t>(allowed - 1));
            }
            const int32_t pixel = path.pixels[static_cast<size_t>(path_index)];
            if (i > 0) {
                oss << ',';
            }
            oss << '[' << (roi.x + trace_x(pixel, roi.w)) << ',' << (roi.y + trace_y(pixel, roi.w)) << ']';
        }
        oss << "]}";
        emitted_points += allowed;
    }

    oss << "],\"path_count\":" << emitted_paths << ",\"point_count\":" << emitted_points << '}';
    *output = oss.str();
    return emitted_paths;
}

// 用于抑制重复模板匹配的交并比。
float rect_iou(const AIRect& a, const AIRect& b) {
    const int32_t x1 = std::max(a.x, b.x);
    const int32_t y1 = std::max(a.y, b.y);
    const int32_t x2 = std::min(a.x + a.w, b.x + b.w);
    const int32_t y2 = std::min(a.y + a.h, b.y + b.h);
    const int32_t iw = std::max<int32_t>(0, x2 - x1);
    const int32_t ih = std::max<int32_t>(0, y2 - y1);
    const int32_t inter = iw * ih;
    if (inter <= 0) {
        return 0.0f;
    }
    const int32_t area_a = a.w * a.h;
    const int32_t area_b = b.w * b.h;
    const int32_t denom = area_a + area_b - inter;
    if (denom <= 0) {
        return 0.0f;
    }
    return static_cast<float>(inter) / static_cast<float>(denom);
}

// 统计参与评分的模板像素数。
// 透明模板匹配会忽略 alpha <= alpha_threshold 的像素。
int32_t scored_pixel_count(const AIImage& templ, bool transparent, int32_t alpha_threshold) {
    if (!transparent) {
        return templ.width * templ.height;
    }

    const int templ_channels = channels_for_format(templ.format);
    int32_t pixels = 0;
    for (int32_t y = 0; y < templ.height; ++y) {
        const uint8_t* templ_row = image_row_ptr(templ, y);
        for (int32_t x = 0; x < templ.width; ++x) {
            const uint8_t* templ_pixel = templ_row + x * templ_channels;
            if (read_alpha(templ_pixel, templ.format) > alpha_threshold) {
                ++pixels;
            }
        }
    }
    return pixels;
}

// 将最小归一化分数转换为可提前退出的绝对差预算。
uint64_t max_diff_for_score(float min_score, int32_t pixels) {
    const double max_diff = (1.0 - static_cast<double>(min_score)) * static_cast<double>(pixels) * 765.0;
    if (max_diff <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(max_diff);
}

// 在图像某个起点处用 BGR 通道绝对差之和给模板评分；
// 当分数已不可能达到 min_score 时提前返回 false。
bool score_template_at(
    const AIImage& image,
    const AIImage& templ,
    int32_t origin_x,
    int32_t origin_y,
    bool transparent,
    int32_t alpha_threshold,
    int32_t channel_tolerance,
    int32_t scored_pixels,
    uint64_t max_allowed_diff,
    float* score) {
    const int image_channels = channels_for_format(image.format);
    const int templ_channels = channels_for_format(templ.format);
    uint64_t total_diff = 0;

    if (!transparent && image.format == templ.format && (image_channels == 3 || image_channels == 4)) {
        const int32_t channels_to_compare = 3;
        for (int32_t y = 0; y < templ.height; ++y) {
            const uint8_t* image_row = image_row_ptr(image, origin_y + y) + origin_x * image_channels;
            const uint8_t* templ_row = image_row_ptr(templ, y);
            for (int32_t x = 0; x < templ.width; ++x) {
                const uint8_t* image_pixel = image_row + x * image_channels;
                const uint8_t* templ_pixel = templ_row + x * templ_channels;
                for (int32_t c = 0; c < channels_to_compare; ++c) {
                    total_diff += static_cast<uint64_t>(std::max(0, std::abs(static_cast<int>(image_pixel[c]) - static_cast<int>(templ_pixel[c])) - channel_tolerance));
                }
                if (total_diff > max_allowed_diff) {
                    return false;
                }
            }
        }

        const double raw_score = 1.0 - static_cast<double>(total_diff) / (static_cast<double>(scored_pixels) * 765.0);
        *score = static_cast<float>(std::max(0.0, std::min(1.0, raw_score)));
        return true;
    }

    if (!transparent && image.format == templ.format && image_channels == 1) {
        for (int32_t y = 0; y < templ.height; ++y) {
            const uint8_t* image_row = image_row_ptr(image, origin_y + y) + origin_x;
            const uint8_t* templ_row = image_row_ptr(templ, y);
            for (int32_t x = 0; x < templ.width; ++x) {
            total_diff += static_cast<uint64_t>(3 * std::max(0, std::abs(static_cast<int>(image_row[x]) - static_cast<int>(templ_row[x])) - channel_tolerance));
                if (total_diff > max_allowed_diff) {
                    return false;
                }
            }
        }

        const double raw_score = 1.0 - static_cast<double>(total_diff) / (static_cast<double>(scored_pixels) * 765.0);
        *score = static_cast<float>(std::max(0.0, std::min(1.0, raw_score)));
        return true;
    }

    for (int32_t y = 0; y < templ.height; ++y) {
        const uint8_t* image_row = image_row_ptr(image, origin_y + y) + origin_x * image_channels;
        const uint8_t* templ_row = image_row_ptr(templ, y);
        for (int32_t x = 0; x < templ.width; ++x) {
            const uint8_t* image_pixel = image_row + x * image_channels;
            const uint8_t* templ_pixel = templ_row + x * templ_channels;
            if (transparent && read_alpha(templ_pixel, templ.format) <= alpha_threshold) {
                continue;
            }

            uint8_t ib = 0;
            uint8_t ig = 0;
            uint8_t ir = 0;
            uint8_t tb = 0;
            uint8_t tg = 0;
            uint8_t tr = 0;
            read_bgr(image_pixel, image.format, &ib, &ig, &ir);
            read_bgr(templ_pixel, templ.format, &tb, &tg, &tr);
            total_diff += static_cast<uint64_t>(std::max(0, std::abs(static_cast<int>(ib) - static_cast<int>(tb)) - channel_tolerance));
            total_diff += static_cast<uint64_t>(std::max(0, std::abs(static_cast<int>(ig) - static_cast<int>(tg)) - channel_tolerance));
            total_diff += static_cast<uint64_t>(std::max(0, std::abs(static_cast<int>(ir) - static_cast<int>(tr)) - channel_tolerance));
            if (total_diff > max_allowed_diff) {
                return false;
            }
        }
    }

    const double raw_score = 1.0 - static_cast<double>(total_diff) / (static_cast<double>(scored_pixels) * 765.0);
    *score = static_cast<float>(std::max(0.0, std::min(1.0, raw_score)));
    return true;
}

// 透明和非透明变体共用的单模板匹配实现。
int32_t find_image_impl(
    const AIImage& image,
    const AIImage& templ,
    bool transparent,
    int32_t alpha_threshold,
    int32_t channel_tolerance,
    float min_score,
    AIImageMatch* output) {
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    *output = AIImageMatch{};
    output->template_index = 0;

    if (!validate_image(image) || !validate_image(templ)) {
        return AI_ERR_IMAGE_FORMAT;
    }
    if (!valid_score(min_score) || alpha_threshold < 0 || alpha_threshold > 255 || channel_tolerance < 0 || channel_tolerance > 255) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (templ.width > image.width || templ.height > image.height) {
        return 0;
    }

    const int32_t scored_pixels = scored_pixel_count(templ, transparent, alpha_threshold);
    if (scored_pixels <= 0) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    const uint64_t max_allowed_diff = max_diff_for_score(min_score, scored_pixels);

    float best_score = -1.0f;
    int32_t best_x = 0;
    int32_t best_y = 0;
    for (int32_t y = 0; y <= image.height - templ.height; ++y) {
        for (int32_t x = 0; x <= image.width - templ.width; ++x) {
            float score = 0.0f;
            if (!score_template_at(image, templ, x, y, transparent, alpha_threshold, channel_tolerance, scored_pixels, max_allowed_diff, &score)) {
                continue;
            }
            if (score > best_score) {
                best_score = score;
                best_x = x;
                best_y = y;
                if (best_score >= 1.0f) {
                    output->box = AIRect{best_x, best_y, templ.width, templ.height};
                    output->score = best_score;
                    output->template_index = 0;
                    return 1;
                }
            }
        }
    }

    if (best_score < min_score) {
        return 0;
    }

    output->box = AIRect{best_x, best_y, templ.width, templ.height};
    output->score = best_score;
    output->template_index = 0;
    return 1;
}

// 共用的多模板匹配实现：收集全部候选、按分数排序，并按模板编号做小范围 IoU 抑制。
int32_t find_images_impl(
    const AIImage& image,
    const AIImage* templates,
    int32_t template_count,
    bool transparent,
    int32_t alpha_threshold,
    int32_t channel_tolerance,
    float min_score,
    AIImageMatch* output,
    int32_t max_output) {
    if (templates == nullptr || output == nullptr || template_count <= 0) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (max_output <= 0) {
        return AI_ERR_BUFFER_TOO_SMALL;
    }
    if (!validate_image(image)) {
        return AI_ERR_IMAGE_FORMAT;
    }
    if (!valid_score(min_score) || alpha_threshold < 0 || alpha_threshold > 255 || channel_tolerance < 0 || channel_tolerance > 255) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    std::vector<AIImageMatch> candidates;
    for (int32_t i = 0; i < template_count; ++i) {
        const AIImage& templ = templates[i];
        if (!validate_image(templ)) {
            return AI_ERR_IMAGE_FORMAT;
        }
        if (templ.width > image.width || templ.height > image.height) {
            continue;
        }

        const int32_t scored_pixels = scored_pixel_count(templ, transparent, alpha_threshold);
        if (scored_pixels <= 0) {
            return AI_ERR_INVALID_ARGUMENT;
        }
        const uint64_t max_allowed_diff = max_diff_for_score(min_score, scored_pixels);

        for (int32_t y = 0; y <= image.height - templ.height; ++y) {
            for (int32_t x = 0; x <= image.width - templ.width; ++x) {
                float score = 0.0f;
                if (!score_template_at(image, templ, x, y, transparent, alpha_threshold, channel_tolerance, scored_pixels, max_allowed_diff, &score)) {
                    continue;
                }
                if (score >= min_score) {
                    AIImageMatch match{};
                    match.box = AIRect{x, y, templ.width, templ.height};
                    match.score = score;
                    match.template_index = i;
                    candidates.push_back(match);
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const AIImageMatch& a, const AIImageMatch& b) {
        return a.score > b.score;
    });

    int32_t count = 0;
    for (const AIImageMatch& candidate : candidates) {
        bool suppressed = false;
        for (int32_t i = 0; i < count; ++i) {
            if (output[i].template_index == candidate.template_index && rect_iou(output[i].box, candidate.box) > 0.30f) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) {
            continue;
        }

        output[count++] = candidate;
        if (count >= max_output) {
            break;
        }
    }

    return count;
}

} // namespace

// 公开 CV 实现：将图像/ROI 转为 GRAY8 字节。
int32_t cv_to_gray(const AIImage& image, const AIRect* roi, uint8_t* output, int32_t output_stride) {
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    AIRect r{};
    const int32_t status = validate_image_and_roi(image, roi, &r);
    if (status != AI_OK) {
        return status;
    }
    if (output_stride < r.w) {
        return AI_ERR_BUFFER_TOO_SMALL;
    }

    const int channels = channels_for_format(image.format);
    if (image.format == AI_IMAGE_GRAY8) {
        for (int32_t y = 0; y < r.h; ++y) {
            const uint8_t* src = image_row_ptr(image, r.y + y) + r.x;
            uint8_t* dst = output + y * output_stride;
            std::memcpy(dst, src, static_cast<size_t>(r.w));
        }
        return r.w * r.h;
    }

    for (int32_t y = 0; y < r.h; ++y) {
        const uint8_t* src = image_row_ptr(image, r.y + y) + r.x * channels;
        uint8_t* dst = output + y * output_stride;
        for (int32_t x = 0; x < r.w; ++x) {
            uint8_t b = 0;
            uint8_t g = 0;
            uint8_t rr = 0;
            read_bgr(src + x * channels, image.format, &b, &g, &rr);
            dst[x] = bgr_to_gray(b, g, rr);
        }
    }

    return r.w * r.h;
}

// 公开 CV 实现：将图像/ROI 阈值化为 0/255 字节。
int32_t cv_threshold(const AIImage& image, const AIRect* roi, int32_t threshold, uint8_t* output, int32_t output_stride) {
    if (output == nullptr || threshold < 0 || threshold > 255) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    AIRect r{};
    const int32_t status = validate_image_and_roi(image, roi, &r);
    if (status != AI_OK) {
        return status;
    }
    if (output_stride < r.w) {
        return AI_ERR_BUFFER_TOO_SMALL;
    }

    const int channels = channels_for_format(image.format);
    for (int32_t y = 0; y < r.h; ++y) {
        const uint8_t* src = image_row_ptr(image, r.y + y) + r.x * channels;
        uint8_t* dst = output + y * output_stride;
        for (int32_t x = 0; x < r.w; ++x) {
            uint8_t b = 0;
            uint8_t g = 0;
            uint8_t rr = 0;
            read_bgr(src + x * channels, image.format, &b, &g, &rr);
            dst[x] = bgr_to_gray(b, g, rr) >= threshold ? 255 : 0;
        }
    }

    return r.w * r.h;
}

// 公开 CV 实现：生成前景骨架路径 JSON。
int32_t cv_extract_trace_json(const AIImage& image, const AIRect* roi, int32_t threshold, bool invert, int32_t max_points, std::string* output) {
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    output->clear();

    AIRect r{};
    const int32_t status = validate_image_and_roi(image, roi, &r);
    if (status != AI_OK) {
        return status;
    }

    std::vector<uint8_t> foreground;
    int32_t actual_threshold = 0;
    int32_t foreground_count = 0;
    const int32_t foreground_status = build_trace_foreground(image, r, threshold, invert, &foreground, &actual_threshold, &foreground_count);
    if (foreground_status != AI_OK) {
        return foreground_status;
    }

    thin_trace_foreground(&foreground, r.w, r.h);
    const int32_t skeleton_count = skeleton_pixel_count(foreground);
    const std::vector<TracePath> paths = extract_trace_paths(foreground, r.w, r.h, r);
    return append_trace_json(paths, r, actual_threshold, invert, max_points, foreground_count, skeleton_count, output);
}

// 公开 CV 实现：计算 ROI 的简单颜色统计。
int32_t cv_mean_color(const AIImage& image, const AIRect* roi, AIColorStats* output) {
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    *output = AIColorStats{};

    AIRect r{};
    const int32_t status = validate_image_and_roi(image, roi, &r);
    if (status != AI_OK) {
        return status;
    }

    const int channels = channels_for_format(image.format);
    uint64_t sum_b = 0;
    uint64_t sum_g = 0;
    uint64_t sum_r = 0;
    int32_t min_gray = 255;
    int32_t max_gray = 0;

    for (int32_t y = 0; y < r.h; ++y) {
        const uint8_t* src = image_row_ptr(image, r.y + y) + r.x * channels;
        for (int32_t x = 0; x < r.w; ++x) {
            uint8_t b = 0;
            uint8_t g = 0;
            uint8_t rr = 0;
            read_bgr(src + x * channels, image.format, &b, &g, &rr);
            sum_b += b;
            sum_g += g;
            sum_r += rr;
            const int32_t gray = bgr_to_gray(b, g, rr);
            min_gray = std::min(min_gray, gray);
            max_gray = std::max(max_gray, gray);
        }
    }

    const int32_t pixels = r.w * r.h;
    output->pixels = pixels;
    output->mean_b = static_cast<double>(sum_b) / pixels;
    output->mean_g = static_cast<double>(sum_g) / pixels;
    output->mean_r = static_cast<double>(sum_r) / pixels;
    output->min_gray = min_gray;
    output->max_gray = max_gray;
    return pixels;
}

// 公开 CV 实现：查找接近目标 BGR 颜色的像素。
int32_t cv_find_color(const AIImage& image, const AIRect* roi, uint32_t target_bgr, int32_t tolerance, AIColorFindResult* output) {
    if (output == nullptr || tolerance < 0 || tolerance > 255) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    *output = AIColorFindResult{};
    output->first_x = -1;
    output->first_y = -1;
    output->bounds = AIRect{0, 0, 0, 0};

    AIRect r{};
    const int32_t status = validate_image_and_roi(image, roi, &r);
    if (status != AI_OK) {
        return status;
    }

    const int32_t target_b = static_cast<int32_t>(target_bgr & 0xffu);
    const int32_t target_g = static_cast<int32_t>((target_bgr >> 8) & 0xffu);
    const int32_t target_r = static_cast<int32_t>((target_bgr >> 16) & 0xffu);

    int32_t min_x = std::numeric_limits<int32_t>::max();
    int32_t min_y = std::numeric_limits<int32_t>::max();
    int32_t max_x = std::numeric_limits<int32_t>::min();
    int32_t max_y = std::numeric_limits<int32_t>::min();

    const int channels = channels_for_format(image.format);
    for (int32_t y = 0; y < r.h; ++y) {
        const int32_t image_y = r.y + y;
        const uint8_t* src = image_row_ptr(image, image_y) + r.x * channels;
        for (int32_t x = 0; x < r.w; ++x) {
            const int32_t image_x = r.x + x;
            uint8_t b = 0;
            uint8_t g = 0;
            uint8_t rr = 0;
            read_bgr(src + x * channels, image.format, &b, &g, &rr);

            if (std::abs(static_cast<int32_t>(b) - target_b) <= tolerance &&
                std::abs(static_cast<int32_t>(g) - target_g) <= tolerance &&
                std::abs(static_cast<int32_t>(rr) - target_r) <= tolerance) {
                if (output->count == 0) {
                    output->first_x = image_x;
                    output->first_y = image_y;
                }
                ++output->count;
                min_x = std::min(min_x, image_x);
                min_y = std::min(min_y, image_y);
                max_x = std::max(max_x, image_x);
                max_y = std::max(max_y, image_y);
            }
        }
    }

    if (output->count > 0) {
        output->bounds = AIRect{min_x, min_y, max_x - min_x + 1, max_y - min_y + 1};
    }
    output->ratio = static_cast<float>(output->count) / static_cast<float>(r.w * r.h);
    return output->count;
}

// 公开 CV 实现：单个不透明模板搜索。
int32_t cv_find_image(const AIImage& image, const AIImage& templ, float min_score, AIImageMatch* output) {
    return find_image_impl(image, templ, false, 255, 0, min_score, output);
}

// 公开 CV 实现：多个不透明模板搜索。
int32_t cv_find_images(const AIImage& image, const AIImage* templates, int32_t template_count, float min_score, AIImageMatch* output, int32_t max_output) {
    return find_images_impl(image, templates, template_count, false, 255, 0, min_score, output, max_output);
}

int32_t cv_find_images_with_tolerance(const AIImage& image, const AIImage* templates, int32_t template_count, int32_t channel_tolerance, float min_score, AIImageMatch* output, int32_t max_output) {
    return find_images_impl(image, templates, template_count, false, 255, channel_tolerance, min_score, output, max_output);
}

// 公开 CV 实现：单个 alpha 感知模板搜索。
int32_t cv_find_transparent_image(const AIImage& image, const AIImage& templ, int32_t alpha_threshold, float min_score, AIImageMatch* output) {
    return find_image_impl(image, templ, true, alpha_threshold, 0, min_score, output);
}

// 公开 CV 实现：多个 alpha 感知模板搜索。
int32_t cv_find_transparent_images(
    const AIImage& image,
    const AIImage* templates,
    int32_t template_count,
    int32_t alpha_threshold,
    float min_score,
    AIImageMatch* output,
    int32_t max_output) {
    return find_images_impl(image, templates, template_count, true, alpha_threshold, 0, min_score, output, max_output);
}

int32_t cv_find_transparent_images_with_tolerance(const AIImage& image, const AIImage* templates, int32_t template_count, int32_t alpha_threshold, int32_t channel_tolerance, float min_score, AIImageMatch* output, int32_t max_output) {
    return find_images_impl(image, templates, template_count, true, alpha_threshold, channel_tolerance, min_score, output, max_output);
}

} // namespace ai
