#pragma once

#include "ai_engine.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ai::coordinate {

inline bool checked_add_i32(int32_t value, int32_t offset, int32_t* output) {
    if (output == nullptr) return false;
    const int64_t shifted = static_cast<int64_t>(value) + offset;
    if (shifted < std::numeric_limits<int32_t>::min() ||
        shifted > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    *output = static_cast<int32_t>(shifted);
    return true;
}

inline bool offset_cv_result(CVMatchResult* result, int32_t origin_x, int32_t origin_y) {
    if (result == nullptr) return false;
    CVMatchResult shifted = *result;
    if (!checked_add_i32(shifted.x, origin_x, &shifted.x) ||
        !checked_add_i32(shifted.y, origin_y, &shifted.y)) {
        return false;
    }
    *result = shifted;
    return true;
}

inline bool offset_ocr_line(AIOcrLine* line, int32_t origin_x, int32_t origin_y) {
    if (line == nullptr) return false;
    AIOcrLine shifted = *line;
    const int64_t center_x = static_cast<int64_t>(shifted.box.x) + shifted.box.w / 2 + origin_x;
    const int64_t center_y = static_cast<int64_t>(shifted.box.y) + shifted.box.h / 2 + origin_y;
    if (center_x < std::numeric_limits<int32_t>::min() ||
        center_x > std::numeric_limits<int32_t>::max() ||
        center_y < std::numeric_limits<int32_t>::min() ||
        center_y > std::numeric_limits<int32_t>::max() ||
        !checked_add_i32(shifted.box.x, origin_x, &shifted.box.x) ||
        !checked_add_i32(shifted.box.y, origin_y, &shifted.box.y)) {
        return false;
    }
    *line = shifted;
    return true;
}

inline bool offset_ocr_lines(std::vector<AIOcrLine>* lines, int32_t origin_x, int32_t origin_y) {
    if (lines == nullptr) return false;
    std::vector<AIOcrLine> shifted = *lines;
    for (AIOcrLine& line : shifted) {
        if (!offset_ocr_line(&line, origin_x, origin_y)) return false;
    }
    *lines = std::move(shifted);
    return true;
}

inline bool offset_ocr_text_result(OCRTextResult* result, int32_t origin_x, int32_t origin_y) {
    if (result == nullptr) return false;
    OCRTextResult shifted = *result;
    if (!checked_add_i32(shifted.x, origin_x, &shifted.x) ||
        !checked_add_i32(shifted.y, origin_y, &shifted.y) ||
        !checked_add_i32(shifted.cx, origin_x, &shifted.cx) ||
        !checked_add_i32(shifted.cy, origin_y, &shifted.cy)) {
        return false;
    }
    *result = shifted;
    return true;
}

inline bool offset_ocr_coord_result(OCRCoordResult* result, int32_t origin_x, int32_t origin_y) {
    if (result == nullptr) return false;
    OCRCoordResult shifted = *result;
    if (!checked_add_i32(shifted.x, origin_x, &shifted.x) ||
        !checked_add_i32(shifted.y, origin_y, &shifted.y)) {
        return false;
    }
    *result = shifted;
    return true;
}

inline bool offset_yolo_boxes(std::vector<AIDetectBox>* boxes, int32_t origin_x, int32_t origin_y) {
    if (boxes == nullptr) return false;
    std::vector<AIDetectBox> shifted = *boxes;
    for (AIDetectBox& box : shifted) {
        const double x1 = static_cast<double>(box.x1) + origin_x;
        const double y1 = static_cast<double>(box.y1) + origin_y;
        const double x2 = static_cast<double>(box.x2) + origin_x;
        const double y2 = static_cast<double>(box.y2) + origin_y;
        const double cx = (static_cast<double>(box.x1) + box.x2) * 0.5 + origin_x;
        const double cy = (static_cast<double>(box.y1) + box.y2) * 0.5 + origin_y;
        const double low = std::numeric_limits<int32_t>::min();
        const double high = std::numeric_limits<int32_t>::max();
        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) ||
            x1 < low || x1 > high || y1 < low || y1 > high ||
            x2 < low || x2 > high || y2 < low || y2 > high ||
            cx < low || cx > high || cy < low || cy > high) {
            return false;
        }
        const float shifted_x1 = static_cast<float>(x1);
        const float shifted_y1 = static_cast<float>(y1);
        const float shifted_x2 = static_cast<float>(x2);
        const float shifted_y2 = static_cast<float>(y2);
        if (!std::isfinite(shifted_x1) || !std::isfinite(shifted_y1) ||
            !std::isfinite(shifted_x2) || !std::isfinite(shifted_y2) ||
            shifted_x1 < low || shifted_x1 > high || shifted_y1 < low || shifted_y1 > high ||
            shifted_x2 < low || shifted_x2 > high || shifted_y2 < low || shifted_y2 > high) {
            return false;
        }
        box.x1 = shifted_x1;
        box.y1 = shifted_y1;
        box.x2 = shifted_x2;
        box.y2 = shifted_y2;
    }
    *boxes = std::move(shifted);
    return true;
}

} // namespace ai::coordinate
