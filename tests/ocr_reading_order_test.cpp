#include "ocr_reading_order.h"

#include <cassert>
#include <vector>

namespace {

using ai::OcrReadingOrderBox;

void require_order(std::vector<OcrReadingOrderBox> boxes, const std::vector<int>& expected_x) {
    ai::sort_ocr_reading_order(&boxes);
    assert(boxes.size() == expected_x.size());
    for (size_t index = 0; index < boxes.size(); ++index) {
        assert(boxes[index].x == expected_x[index]);
    }
}

}  // namespace

int main() {
    // Business regression: the left number is slightly lower than the long text box.
    require_order({{16, 7, 463, 22}, {0, 11, 21, 14}}, {0, 16});

    // Three independent rows remain top-to-bottom.
    require_order({{0, 58, 479, 19}, {2, 5, 66, 20}, {1, 31, 310, 19}}, {2, 1, 0});

    // A small punctuation box whose center follows the baseline belongs to the same row.
    require_order({{40, 10, 100, 24}, {25, 16, 6, 10}, {0, 11, 20, 20}}, {0, 25, 40});

    // Adjacent rows without sufficient overlap must not be merged.
    require_order({{100, 30, 20, 12}, {0, 8, 80, 18}, {0, 29, 80, 18}}, {0, 0, 100});

    // Exact geometry ties retain detector order.
    std::vector<OcrReadingOrderBox> tied{{10, 10, 20, 10}, {10, 10, 30, 10}};
    ai::sort_ocr_reading_order(&tied);
    assert(tied[0].w == 20 && tied[1].w == 30);

    // Four compact detector fragments from one baseline become one crop.
    bool merged = false;
    size_t fragment_count = 0;
    auto joined = ai::merge_fragmented_ocr_boxes(
        {{0, 2, 19, 21}, {25, 7, 22, 15}, {45, 7, 17, 16}, {65, 3, 26, 22}},
        0.25f,
        &merged,
        &fragment_count);
    assert(merged && fragment_count == 4 && joined.size() == 1);
    assert(joined[0].x == 0 && joined[0].y == 2 &&
           joined[0].w == 91 && joined[0].h == 23);

    // Two normal adjacent labels are not joined merely because padding would
    // overlap. This also protects legitimate repeated text in separate boxes.
    joined = ai::merge_fragmented_ocr_boxes(
        {{0, 10, 20, 18}, {24, 10, 20, 18}}, 0.25f, &merged, &fragment_count);
    assert(!merged && joined.size() == 2);

    // Three compact but non-overlapping labels are still independent. A
    // padded-crop overlap alone is not sufficient evidence for joining.
    joined = ai::merge_fragmented_ocr_boxes(
        {{0, 10, 20, 18}, {24, 10, 20, 18}, {48, 10, 20, 18}},
        0.25f,
        &merged,
        &fragment_count);
    assert(!merged && joined.size() == 3);

    // Actual detector overlap is reliable geometric evidence for one crop.
    joined = ai::merge_fragmented_ocr_boxes(
        {{0, 10, 30, 18}, {28, 10, 30, 18}}, 0.25f, &merged, &fragment_count);
    assert(merged && joined.size() == 1 && joined[0].w == 58);

    // Distant labels and different rows always remain independent.
    joined = ai::merge_fragmented_ocr_boxes(
        {{0, 10, 20, 18}, {100, 10, 20, 18}, {0, 40, 20, 18}},
        0.25f,
        &merged,
        &fragment_count);
    assert(!merged && joined.size() == 3);
    return 0;
}
