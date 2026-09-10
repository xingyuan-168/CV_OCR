#include "ocr_color_filter.h"
#include "backends.h"

#include <cassert>
#include <algorithm>
#include <cstring>
#include <iostream>

int main() {
    ai::OcrColorFilter filter;
    std::string error;

    assert(ai::parse_ocr_color_filter(nullptr, &filter, &error));
    assert(filter.automatic && filter.rules.empty());
    assert(ai::parse_ocr_color_filter("  ", &filter, &error));
    assert(filter.automatic && filter.rules.empty());

    assert(ai::parse_ocr_color_filter(
        "ff00ff-101010|00FFFF-202020", &filter, &error));
    assert(!filter.automatic && filter.rules.size() == 2);
    assert(filter.rules[0].r == 0xff && filter.rules[0].g == 0x00 &&
           filter.rules[0].b == 0xff && filter.rules[0].tolerance_r == 0x10);
    assert(filter.rules[1].r == 0x00 && filter.rules[1].g == 0xff &&
           filter.rules[1].b == 0xff && filter.rules[1].tolerance_b == 0x20);

    for (const char* invalid : {
             "FF00FF-10101",
             "FF00FF101010",
             "FF00FG-101010",
             "|FF00FF-101010",
             "FF00FF-101010|",
             "FF00FF-101010||00FFFF-202020"}) {
        error.clear();
        assert(!ai::parse_ocr_color_filter(invalid, &filter, &error));
        assert(!error.empty());
    }

    std::string too_many;
    for (int i = 0; i < 17; ++i) {
        if (!too_many.empty()) too_many += '|';
        too_many += "FFFFFF-202020";
    }
    assert(!ai::parse_ocr_color_filter(too_many.c_str(), &filter, &error));
    assert(!error.empty());

    std::vector<uint8_t> pixels(100u * 80u * 3u, 32);
    AIImage image{pixels.data(), 100, 80, 300, AI_IMAGE_BGR24};
    assert(ai::parse_ocr_color_filter(nullptr, &filter, &error));
    const auto candidates = ai::make_ocr_filter_candidates(image, filter);
    assert(candidates.size() >= 3);
    assert(std::string(candidates.front().mode) == "original");
    assert(candidates.front().width == 100 && candidates.front().height == 80);
    assert(std::string(candidates[1].mode) == "bgr_scaled");
    assert(candidates[1].width == 300 && candidates[1].height == 240);

    // Explicit rules keep the exact mask and add generic antialias recovery
    // candidates. No business text, filename or image coordinate is involved.
    std::vector<uint8_t> antialias_pixels(20u * 10u * 3u, 20);
    for (int y = 2; y < 8; ++y) {
        for (int x = 2; x < 18; ++x) {
            const uint8_t value = (x == 2 || x == 17) ? 207 : 255;
            uint8_t* pixel = antialias_pixels.data() +
                (static_cast<size_t>(y) * 20u + x) * 3u;
            pixel[0] = pixel[1] = pixel[2] = value;
        }
    }
    AIImage antialias_image{
        antialias_pixels.data(), 20, 10, 60, AI_IMAGE_BGR24};
    assert(ai::parse_ocr_color_filter("FFFFFF-202020", &filter, &error));
    const auto antialias_candidates =
        ai::make_ocr_filter_candidates(antialias_image, filter);
    const auto strict = std::find_if(
        antialias_candidates.begin(),
        antialias_candidates.end(),
        [](const auto& candidate) {
            return std::string(candidate.mode) == "color_soft_scaled";
        });
    const auto recovered = std::find_if(
        antialias_candidates.begin(),
        antialias_candidates.end(),
        [](const auto& candidate) {
            return std::string(candidate.mode) ==
                "color_antialias_soft_scaled";
        });
    assert(strict != antialias_candidates.end());
    assert(recovered != antialias_candidates.end());
    assert(recovered->foreground_pixels > strict->foreground_pixels);

    // The same recovery rule works for dark antialiased text on a bright
    // background; it is not tied to white game text.
    std::vector<uint8_t> dark_pixels(40u * 20u * 3u, 240);
    for (int y = 4; y < 16; ++y) {
        for (int x = 4; x < 36; ++x) {
            const uint8_t value = (x == 4 || x == 35) ? 18 : 0;
            uint8_t* pixel = dark_pixels.data() +
                (static_cast<size_t>(y) * 40u + x) * 3u;
            pixel[0] = pixel[1] = pixel[2] = value;
        }
    }
    AIImage dark_image{dark_pixels.data(), 40, 20, 120, AI_IMAGE_BGR24};
    assert(ai::parse_ocr_color_filter("000000-101010", &filter, &error));
    const auto dark_candidates = ai::make_ocr_filter_candidates(dark_image, filter);
    const auto dark_strict = std::find_if(
        dark_candidates.begin(), dark_candidates.end(), [](const auto& candidate) {
            return std::string(candidate.mode) == "color_soft_scaled";
        });
    const auto dark_recovered = std::find_if(
        dark_candidates.begin(), dark_candidates.end(), [](const auto& candidate) {
            return std::string(candidate.mode) ==
                "color_antialias_soft_scaled";
        });
    assert(dark_strict != dark_candidates.end());
    assert(dark_recovered != dark_candidates.end());
    assert(dark_recovered->foreground_pixels > dark_strict->foreground_pixels);

    // Automatic preprocessing remains bounded and valid for empty/pure-color
    // inputs, while explicit colorful masks preserve their target support.
    std::vector<uint8_t> color_pixels(48u * 24u * 3u, 48);
    for (int y = 5; y < 19; ++y) {
        for (int x = 6; x < 42; ++x) {
            uint8_t* pixel = color_pixels.data() +
                (static_cast<size_t>(y) * 48u + x) * 3u;
            pixel[0] = 0x20;
            pixel[1] = 0xA0;
            pixel[2] = 0xF0;
        }
    }
    AIImage color_image{color_pixels.data(), 48, 24, 144, AI_IMAGE_BGR24};
    assert(ai::parse_ocr_color_filter("F0A020-101010", &filter, &error));
    const auto color_candidates = ai::make_ocr_filter_candidates(color_image, filter);
    assert(std::any_of(
        color_candidates.begin(), color_candidates.end(), [](const auto& candidate) {
            return !candidate.foreground_mask.empty() &&
                candidate.foreground_pixels > 0;
        }));

    std::vector<uint8_t> pure_pixels(32u * 16u * 3u, 128);
    AIImage pure_image{pure_pixels.data(), 32, 16, 96, AI_IMAGE_BGR24};
    assert(ai::parse_ocr_color_filter(nullptr, &filter, &error));
    const auto pure_candidates = ai::make_ocr_filter_candidates(pure_image, filter);
    assert(!pure_candidates.empty());
    assert(std::string(pure_candidates.front().mode) == "original");

    std::vector<AIOcrLine> lines(1);
    lines[0].box = AIRect{5, 5, 60, 18};
    lines[0].confidence = 0.95f;
    std::strcpy(lines[0].text, u8"法宝合成");
    auto evaluation = ai::evaluate_ocr_candidate(lines, 100, 80, 0.5f);
    assert(!evaluation.suspicious && evaluation.strong);
    assert(evaluation.unicode_chars == 4);
    assert(evaluation.high_confidence_chars == 4);

    lines[0].box = AIRect{0, 0, 100, 80};
    lines[0].confidence = 0.1f;
    evaluation = ai::evaluate_ocr_candidate(lines, 100, 80, 0.0f);
    assert(evaluation.suspicious);

    lines.clear();
    evaluation = ai::evaluate_ocr_candidate(lines, 100, 80, 0.0f);
    assert(evaluation.suspicious && !evaluation.spatial_complete);
    assert(evaluation.foreground_column_coverage == 0.0f);
    lines.resize(1);

    lines[0].box = AIRect{15, 12, 90, 45};
    ai::OcrFilterCandidate scaled_candidate;
    scaled_candidate.scale_x = 3.0f;
    scaled_candidate.scale_y = 3.0f;
    ai::remap_ocr_lines_from_candidate(&lines, scaled_candidate, 100, 80);
    assert(lines[0].box.x == 5 && lines[0].box.y == 4);
    assert(lines[0].box.w == 30 && lines[0].box.h == 15);

    ai::OcrTargetGroups targets{{u8"法宝"}, {u8"金钱镖"}};
    lines[0].box = AIRect{5, 5, 60, 18};
    lines[0].confidence = 0.95f;
    std::strcpy(lines[0].text, u8"法宝合成");
    evaluation = ai::evaluate_ocr_candidate(lines, 100, 80, 0.5f, &targets);
    assert(evaluation.target_groups_matched == 1);

    // High confidence cannot certify a result whose boxes miss source text at
    // the right edge.
    ai::OcrFilterCandidate support;
    support.width = 100;
    support.height = 20;
    support.foreground_mask.assign(2000, 0);
    for (int y = 2; y < 18; ++y) {
        for (int x = 5; x < 95; ++x) {
            support.foreground_mask[static_cast<size_t>(y) * 100 + x] = 1;
        }
    }
    support.active_columns = 90;
    lines.resize(1);
    lines[0].box = AIRect{5, 2, 55, 16};
    lines[0].confidence = 0.99f;
    std::strcpy(lines[0].text, "abcdef");
    evaluation = ai::evaluate_ocr_candidate(
        lines, 100, 20, 0.5f, nullptr, &support, nullptr);
    assert(!evaluation.spatial_complete);
    assert(evaluation.foreground_column_coverage < 0.70f);
    assert(evaluation.uncovered_trailing_columns >= 30);
    assert(evaluation.suspicious && !evaluation.strong);

    const auto incomplete = evaluation;
    lines[0].box = AIRect{5, 1, 90, 18};
    std::strcpy(lines[0].text, "abcdef");
    lines[0].confidence = 0.99f;
    support.foreground_x0 = 5;
    support.foreground_y0 = 2;
    support.foreground_x1 = 95;
    support.foreground_y1 = 18;
    support.foreground_pixels = 90u * 16u;
    evaluation = ai::evaluate_ocr_candidate(
        lines, 100, 20, 0.5f, nullptr, &support, nullptr);
    assert(evaluation.spatial_complete);
    assert(ai::is_better_ocr_candidate(evaluation, incomplete));

    // A left-anchored small line with a large unexplained tail cannot use the
    // automatic one-inference fast path before source support is inspected.
    lines[0].box = AIRect{0, 1, 60, 18};
    lines[0].confidence = 0.99f;
    std::strcpy(lines[0].text, "abcdef");
    evaluation = ai::evaluate_ocr_candidate(lines, 100, 20, 0.5f);
    assert(evaluation.suspicious && !evaluation.strong);

    // Legitimate repeated characters remain untouched; no string-level
    // duplicate removal participates in candidate scoring.
    lines[0].box = AIRect{2, 2, 96, 16};
    lines[0].confidence = 0.99f;
    std::strcpy(lines[0].text, u8"人人宫宫");
    evaluation = ai::evaluate_ocr_candidate(lines, 100, 20, 0.5f);
    assert(evaluation.unicode_chars == 4 && evaluation.strong);

    // Regardless of how many preprocessing forms are available, inference is
    // capped at one fast candidate plus four fallbacks.
    int callback_calls = 0;
    std::vector<AIOcrLine> pipeline_output;
    const auto empty_recognizer = [&callback_calls](
        const AIImage&,
        std::vector<AIOcrLine>* recognized,
        ai::OcrRecognitionDiagnostics*,
        std::string*) -> int32_t {
        ++callback_calls;
        recognized->clear();
        return 0;
    };
    assert(ai::parse_ocr_color_filter("FFFFFF-202020", &filter, &error));
    assert(ai::run_ocr_candidate_pipeline(
        antialias_image, filter, 0.0f, nullptr, empty_recognizer,
        &pipeline_output, &error) == 0);
    assert(callback_calls <= 5);
    callback_calls = 0;
    assert(ai::parse_ocr_color_filter(nullptr, &filter, &error));
    assert(ai::run_ocr_candidate_pipeline(
        pure_image, filter, 0.0f, nullptr, empty_recognizer,
        &pipeline_output, &error) == 0);
    assert(callback_calls <= 5);

    // Joint-line recognition must not replace the established split result
    // unconditionally. The backend exposes both hypotheses internally and the
    // pipeline applies the same generic scoring rules to each of them.
    std::vector<uint8_t> hypothesis_pixels(100u * 20u * 3u, 128);
    AIImage hypothesis_image{
        hypothesis_pixels.data(), 100, 20, 300, AI_IMAGE_BGR24};
    const auto dual_hypothesis_recognizer = [](
        const AIImage&,
        std::vector<AIOcrLine>* recognized,
        ai::OcrRecognitionDiagnostics* diagnostics,
        std::string*) -> int32_t {
        recognized->assign(1, AIOcrLine{});
        recognized->front().box = AIRect{0, 1, 60, 18};
        recognized->front().confidence = 0.99f;
        std::strcpy(recognized->front().text, "merged");
        diagnostics->detected_box_count = 3;
        diagnostics->used_merged_line_recognition = true;
        diagnostics->split_box_lines.assign(3, AIOcrLine{});
        for (size_t i = 0; i < diagnostics->split_box_lines.size(); ++i) {
            AIOcrLine& line = diagnostics->split_box_lines[i];
            line.box = AIRect{static_cast<int32_t>(i * 35), 6, 30, 8};
            line.confidence = 0.95f;
            std::strcpy(line.text, i == 0 ? "A" : (i == 1 ? "B" : "C"));
        }
        return 1;
    };
    ai::OcrPipelineSelection dual_selection;
    assert(ai::parse_ocr_color_filter(nullptr, &filter, &error));
    assert(ai::run_ocr_candidate_pipeline(
        hypothesis_image,
        filter,
        0.0f,
        nullptr,
        dual_hypothesis_recognizer,
        &pipeline_output,
        &error,
        &dual_selection) == 3);
    assert(pipeline_output.size() == 3);
    assert(std::strcmp(pipeline_output[0].text, "A") == 0);
    assert(std::strcmp(pipeline_output[1].text, "B") == 0);
    assert(std::strcmp(pipeline_output[2].text, "C") == 0);
    assert(!dual_selection.used_merged_line_recognition);
    assert(dual_selection.attempts.size() == 2);
    assert(dual_selection.attempts[0].candidate_type == "original/merged");
    assert(!dual_selection.attempts[0].selected);
    assert(!dual_selection.attempts[0].rejection_reason.empty());
    assert(dual_selection.attempts[1].candidate_type == "original/split");
    assert(dual_selection.attempts[1].selected);
    assert(dual_selection.attempts[1].rejection_reason.empty());

    lines.push_back(AIOcrLine{});
    lines.back().box = AIRect{5, 30, 60, 18};
    lines.back().confidence = 0.1f;
    std::strcpy(lines.back().text, u8"低分");
    ai::filter_ocr_lines_by_confidence(&lines, 0.5f);
    assert(lines.size() == 1);
    std::cout << "OCR color filter parser passed\n";
    return 0;
}
