#include "ocr_reading_order.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace ai {
namespace {

int median(std::vector<int> values) {
    if (values.empty()) return 0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if ((values.size() & 1u) != 0u) return values[middle];
    const int upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return (values[middle - 1] + upper) / 2;
}

struct TextLine {
    std::vector<OcrReadingOrderBox> boxes;
    int center_y2 = 0;
    int height = 1;
    int top = 0;
    size_t first_detection_index = 0;

    void refresh() {
        std::vector<int> centers;
        std::vector<int> heights;
        centers.reserve(boxes.size());
        heights.reserve(boxes.size());
        top = std::numeric_limits<int>::max();
        first_detection_index = std::numeric_limits<size_t>::max();
        for (const auto& box : boxes) {
            centers.push_back(2 * box.y + box.h);
            heights.push_back(std::max(1, box.h));
            top = std::min(top, box.y);
            first_detection_index = std::min(first_detection_index, box.detection_index);
        }
        center_y2 = median(std::move(centers));
        height = std::max(1, median(std::move(heights)));
    }
};

struct LineFit {
    bool matches = false;
    float overlap_ratio = 0.0f;
    int center_distance2 = std::numeric_limits<int>::max();
};

LineFit fit_line(const OcrReadingOrderBox& box, const TextLine& line) {
    LineFit fit;
    const int box_height = std::max(1, box.h);
    const int line_top = (line.center_y2 - line.height) / 2;
    const int line_bottom = line_top + line.height;
    const int box_bottom = box.y + box_height;
    const int overlap = std::max(0, std::min(box_bottom, line_bottom) - std::max(box.y, line_top));
    fit.overlap_ratio = static_cast<float>(overlap) / static_cast<float>(std::min(box_height, line.height));
    fit.center_distance2 = std::abs((2 * box.y + box_height) - line.center_y2);
    const float center_tolerance2 = 0.70f * static_cast<float>(std::min(box_height, line.height));
    fit.matches = fit.overlap_ratio >= 0.50f ||
        static_cast<float>(fit.center_distance2) <= center_tolerance2;
    return fit;
}

}  // namespace

std::vector<OcrReadingOrderLine> group_ocr_reading_order(
    const std::vector<OcrReadingOrderBox>& boxes) {
    if (boxes.empty()) return {};
    std::vector<OcrReadingOrderBox> indexed = boxes;
    for (size_t index = 0; index < indexed.size(); ++index) {
        indexed[index].detection_index = index;
    }

    std::vector<OcrReadingOrderBox> pending = std::move(indexed);
    std::stable_sort(pending.begin(), pending.end(), [](const auto& left, const auto& right) {
        const int left_center_y2 = 2 * left.y + left.h;
        const int right_center_y2 = 2 * right.y + right.h;
        if (left_center_y2 != right_center_y2) return left_center_y2 < right_center_y2;
        if (left.x != right.x) return left.x < right.x;
        return left.detection_index < right.detection_index;
    });

    std::vector<TextLine> lines;
    for (const auto& box : pending) {
        size_t best_line = lines.size();
        LineFit best_fit;
        for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
            const LineFit fit = fit_line(box, lines[line_index]);
            if (!fit.matches) continue;
            if (best_line == lines.size() || fit.overlap_ratio > best_fit.overlap_ratio ||
                (fit.overlap_ratio == best_fit.overlap_ratio && fit.center_distance2 < best_fit.center_distance2)) {
                best_line = line_index;
                best_fit = fit;
            }
        }
        if (best_line == lines.size()) {
            TextLine line;
            line.boxes.push_back(box);
            line.refresh();
            lines.push_back(std::move(line));
        } else {
            lines[best_line].boxes.push_back(box);
            lines[best_line].refresh();
        }
    }

    std::stable_sort(lines.begin(), lines.end(), [](const TextLine& left, const TextLine& right) {
        if (left.center_y2 != right.center_y2) return left.center_y2 < right.center_y2;
        if (left.top != right.top) return left.top < right.top;
        return left.first_detection_index < right.first_detection_index;
    });

    std::vector<OcrReadingOrderLine> output;
    output.reserve(lines.size());
    for (auto& line : lines) {
        std::stable_sort(line.boxes.begin(), line.boxes.end(), [](const auto& left, const auto& right) {
            if (left.x != right.x) return left.x < right.x;
            if (left.y != right.y) return left.y < right.y;
            return left.detection_index < right.detection_index;
        });
        output.push_back(std::move(line.boxes));
    }
    return output;
}

std::vector<OcrReadingOrderBox> merge_fragmented_ocr_boxes(
    const std::vector<OcrReadingOrderBox>& boxes,
    float horizontal_padding_ratio,
    bool* merged_any,
    size_t* merged_fragment_count) {
    if (merged_any != nullptr) *merged_any = false;
    if (merged_fragment_count != nullptr) *merged_fragment_count = 0;
    if (boxes.size() < 2) return boxes;

    const auto lines = group_ocr_reading_order(boxes);
    std::vector<OcrReadingOrderBox> output;
    output.reserve(boxes.size());
    for (const auto& line : lines) {
        if (line.size() < 2) {
            output.insert(output.end(), line.begin(), line.end());
            continue;
        }

        std::vector<int> heights;
        heights.reserve(line.size());
        for (const auto& box : line) heights.push_back(std::max(1, box.h));
        const int line_height = std::max(1, median(std::move(heights)));
        const int cluster_gap_limit = std::max(
            2, static_cast<int>(std::ceil(line_height * 1.5f)));

        size_t begin = 0;
        while (begin < line.size()) {
            size_t end = begin + 1;
            bool padded_overlap = false;
            bool geometry_overlap = false;
            int cluster_left = line[begin].x;
            int cluster_right = line[begin].x + line[begin].w;
            int width_sum = line[begin].w;
            while (end < line.size()) {
                const auto& previous = line[end - 1];
                const auto& next = line[end];
                const int gap = next.x - (previous.x + previous.w);
                if (gap > cluster_gap_limit) break;
                const int previous_padding = std::max(
                    1, static_cast<int>(std::ceil(
                           std::max(1, previous.h) * horizontal_padding_ratio)));
                const int next_padding = std::max(
                    1, static_cast<int>(std::ceil(
                           std::max(1, next.h) * horizontal_padding_ratio)));
                if (gap < previous_padding + next_padding) padded_overlap = true;
                if (gap < 0) geometry_overlap = true;
                cluster_left = std::min(cluster_left, next.x);
                cluster_right = std::max(cluster_right, next.x + next.w);
                width_sum += next.w;
                ++end;
            }

            const size_t fragment_count = end - begin;
            const int span = std::max(1, cluster_right - cluster_left);
            const float compactness = static_cast<float>(width_sum) / span;
            int overlapping_pairs = 0;
            for (size_t i = begin + 1; i < end; ++i) {
                if (line[i].x < line[i - 1].x + line[i - 1].w) {
                    ++overlapping_pairs;
                }
            }
            // Two boxes are joined only when their recognition crops overlap.
            // Three or more compact fragments are also treated as a split line.
            const bool should_merge = fragment_count >= 2 &&
                ((fragment_count == 2 && geometry_overlap) ||
                 (fragment_count >= 3 &&
                  (overlapping_pairs >= 2 ||
                   (geometry_overlap && padded_overlap &&
                    compactness >= 0.85f))));
            if (!should_merge) {
                output.insert(
                    output.end(), line.begin() + begin, line.begin() + end);
                begin = end;
                continue;
            }

            OcrReadingOrderBox joined = line[begin];
            int top = joined.y;
            int right = joined.x + joined.w;
            int bottom = joined.y + joined.h;
            for (size_t i = begin + 1; i < end; ++i) {
                joined.x = std::min(joined.x, line[i].x);
                top = std::min(top, line[i].y);
                right = std::max(right, line[i].x + line[i].w);
                bottom = std::max(bottom, line[i].y + line[i].h);
                joined.detection_index = std::min(
                    joined.detection_index, line[i].detection_index);
            }
            joined.y = top;
            joined.w = std::max(1, right - joined.x);
            joined.h = std::max(1, bottom - joined.y);
            output.push_back(joined);
            if (merged_any != nullptr) *merged_any = true;
            if (merged_fragment_count != nullptr) {
                *merged_fragment_count += fragment_count;
            }
            begin = end;
        }
    }
    return output;
}

void sort_ocr_reading_order(std::vector<OcrReadingOrderBox>* boxes) {
    if (boxes == nullptr || boxes->size() < 2) return;
    const auto lines = group_ocr_reading_order(*boxes);
    boxes->clear();
    for (const auto& line : lines) {
        boxes->insert(boxes->end(), line.begin(), line.end());
    }
}

}  // namespace ai
