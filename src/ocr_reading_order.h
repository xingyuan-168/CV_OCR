#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ai {

struct OcrReadingOrderBox {
    int32_t x = 0;
    int32_t y = 0;
    int32_t w = 0;
    int32_t h = 0;
    size_t detection_index = 0;
};

using OcrReadingOrderLine = std::vector<OcrReadingOrderBox>;

// Groups boxes into reading-order lines without modifying their geometry.
// Lines are top-to-bottom and boxes inside a line are left-to-right.
std::vector<OcrReadingOrderLine> group_ocr_reading_order(
    const std::vector<OcrReadingOrderBox>& boxes);

// Conservatively joins detector fragments that are very likely to belong to
// one recognition crop. Distant labels and different rows remain separate.
std::vector<OcrReadingOrderBox> merge_fragmented_ocr_boxes(
    const std::vector<OcrReadingOrderBox>& boxes,
    float horizontal_padding_ratio,
    bool* merged_any = nullptr,
    size_t* merged_fragment_count = nullptr);

// Groups OCR detection boxes into text lines, then orders lines top-to-bottom
// and boxes in each line left-to-right. Geometry is not modified.
void sort_ocr_reading_order(std::vector<OcrReadingOrderBox>* boxes);

}  // namespace ai
