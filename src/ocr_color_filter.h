#pragma once

#include "ai_engine.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ai {

struct OcrRecognitionDiagnostics;

struct OcrColorRule {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t tolerance_r = 0;
    uint8_t tolerance_g = 0;
    uint8_t tolerance_b = 0;
};

struct OcrColorFilter {
    bool automatic = true;
    std::vector<OcrColorRule> rules;
    std::string canonical;
};

struct OcrFilterCandidate {
    std::vector<uint8_t> pixels;
    int32_t width = 0;
    int32_t height = 0;
    int32_t stride = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    const char* mode = "original";
    // Sparse foreground support in candidate coordinates. Empty support means
    // that completeness must be inferred from recognition geometry only.
    std::vector<uint8_t> foreground_mask;
    int32_t foreground_x0 = 0;
    int32_t foreground_y0 = 0;
    int32_t foreground_x1 = 0;
    int32_t foreground_y1 = 0;
    size_t foreground_pixels = 0;
    size_t active_columns = 0;

    AIImage view() const {
        return AIImage{
            const_cast<uint8_t*>(pixels.data()), width, height, stride, AI_IMAGE_BGR24};
    }
};

using OcrTargetGroups = std::vector<std::vector<std::string>>;

struct OcrCandidateEvaluation {
    double quality = 0.0;
    size_t unicode_chars = 0;
    size_t high_confidence_chars = 0;
    size_t plausible_lines = 0;
    size_t target_groups_matched = 0;
    float mean_confidence = 0.0f;
    float foreground_column_coverage = 1.0f;
    float padded_overlap_ratio = 0.0f;
    size_t uncovered_leading_columns = 0;
    size_t uncovered_trailing_columns = 0;
    size_t fragmented_pairs = 0;
    bool spatial_complete = true;
    bool suspicious = true;
    bool strong = false;
};

struct OcrCandidateTrace {
    std::string candidate_type;
    int32_t status = 0;
    float foreground_column_coverage = 0.0f;
    float padded_overlap_ratio = 0.0f;
    float mean_confidence = 0.0f;
    size_t fragmented_pairs = 0;
    size_t unicode_chars = 0;
    bool spatial_complete = false;
    bool used_eight_connectivity = false;
    bool used_merged_line_recognition = false;
    bool selected = false;
    int64_t detection_us = -1;
    int64_t recognition_us = -1;
    int64_t postprocess_us = -1;
    std::string rejection_reason;
};

// Internal-only metadata for keeping compatibility TEXT/JSON/Find semantics
// on the same selected preprocessing and detection hypothesis.
struct OcrPipelineSelection {
    std::string candidate_type;
    float foreground_column_coverage = 1.0f;
    float padded_overlap_ratio = 0.0f;
    bool spatial_complete = true;
    bool used_eight_connectivity = false;
    bool used_merged_line_recognition = false;
    std::vector<OcrCandidateTrace> attempts;
};

bool parse_ocr_color_filter(const char* text, OcrColorFilter* output, std::string* error);

std::vector<OcrFilterCandidate> make_ocr_filter_candidates(
    const AIImage& source,
    const OcrColorFilter& filter);

void remap_ocr_lines_from_candidate(
    std::vector<AIOcrLine>* lines,
    const OcrFilterCandidate& candidate,
    int32_t original_width,
    int32_t original_height);

OcrCandidateEvaluation evaluate_ocr_candidate(
    const std::vector<AIOcrLine>& lines,
    int32_t image_width,
    int32_t image_height,
    float min_confidence,
    const OcrTargetGroups* target_groups = nullptr,
    const OcrFilterCandidate* source_candidate = nullptr,
    const OcrRecognitionDiagnostics* diagnostics = nullptr);

bool is_better_ocr_candidate(
    const OcrCandidateEvaluation& candidate,
    const OcrCandidateEvaluation& current_best);

void filter_ocr_lines_by_confidence(
    std::vector<AIOcrLine>* lines,
    float min_confidence);

using OcrRecognizeCallback = std::function<int32_t(
    const AIImage&,
    std::vector<AIOcrLine>*,
    OcrRecognitionDiagnostics*,
    std::string*)>;

int32_t run_ocr_candidate_pipeline(
    const AIImage& source,
    const OcrColorFilter& filter,
    float min_confidence,
    const OcrTargetGroups* target_groups,
    const OcrRecognizeCallback& recognize,
    std::vector<AIOcrLine>* output,
    std::string* error,
    OcrPipelineSelection* selection = nullptr);

} // namespace ai
