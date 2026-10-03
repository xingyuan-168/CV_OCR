#pragma once
#include "yolo_preprocess.h"
#include <numeric>
#include <cstring>
#include <limits>
#include <string>
namespace ai {
inline float compute_iou(float x1a, float y1a, float x2a, float y2a,
                  float x1b, float y1b, float x2b, float y2b) {
    const float inter_x1 = std::max(x1a, x1b);
    const float inter_y1 = std::max(y1a, y1b);
    const float inter_x2 = std::min(x2a, x2b);
    const float inter_y2 = std::min(y2a, y2b);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float area_a = (x2a - x1a) * (y2a - y1a);
    const float area_b = (x2b - x1b) * (y2b - y1b);
    const float union_area = area_a + area_b - inter_area;
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

// NMS 和公开标签复制前的内部 YOLO 候选检测框。
struct RawDetection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// 贪心 NMS：按分数降序排序，并抑制重叠框。
inline std::vector<int> nms_greedy(const std::vector<RawDetection>& dets, float iou_threshold) {
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int> keep;
    keep.reserve(dets.size());

    for (int idx : indices) {
        if (suppressed[idx]) {
            continue;
        }
        keep.push_back(idx);
        for (size_t j = 0; j < indices.size(); ++j) {
            const int other = indices[j];
            if (suppressed[other] || other == idx) {
                continue;
            }
            const float iou = compute_iou(
                dets[idx].x1, dets[idx].y1, dets[idx].x2, dets[idx].y2,
                dets[other].x1, dets[other].y1, dets[other].x2, dets[other].y2);
            if (iou > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return keep;
}


inline int32_t yolo_decode(const float* raw_output, int64_t attrs, int64_t count,
    const AIImage& image, const YoloLetterbox& lbox, float conf_threshold, float nms_threshold,
    const std::vector<std::string>& labels, std::vector<RawDetection>& candidates, std::vector<AIDetectBox>* output) {
        const int64_t num_attrs = attrs;      // e.g. 84 = 4 + 80
        const int64_t num_candidates = count; // e.g. 8400
        if (!raw_output || num_attrs <= 4 || num_attrs > std::numeric_limits<int32_t>::max()) {
            return AI_ERR_RUNTIME;
        }
        const int num_classes = static_cast<int>(num_attrs - 4);
        // 按候选框遍历 [1, 84, 8400] 输出。
        // raw_output 布局为 attr_i * num_candidates + candidate_j。
        if (num_candidates < 0 || static_cast<uint64_t>(num_candidates) > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            return AI_ERR_RUNTIME;
        }
        candidates.clear();
        candidates.reserve(static_cast<size_t>(num_candidates));

        for (int64_t j = 0; j < num_candidates; ++j) {
            // 查找最佳类别分数。
            float max_score = 0.0f;
            int best_class = 0;
            for (int c = 0; c < num_classes; ++c) {
                const float s = raw_output[(4 + c) * num_candidates + j];
                if (s > max_score) {
                    max_score = s;
                    best_class = c;
                }
            }

            if (max_score < conf_threshold) {
                continue;
            }

            // 输入空间中的 cx、cy、w、h（例如带 letterbox 的 640x640）。
            const float cx = raw_output[0 * num_candidates + j];
            const float cy = raw_output[1 * num_candidates + j];
            const float w  = raw_output[2 * num_candidates + j];
            const float h  = raw_output[3 * num_candidates + j];

            // 转换为输入空间中的 x1,y1,x2,y2。
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            // 映射回原图坐标。
            x1 = (x1 - lbox.pad_x) / lbox.scale;
            y1 = (y1 - lbox.pad_y) / lbox.scale;
            x2 = (x2 - lbox.pad_x) / lbox.scale;
            y2 = (y2 - lbox.pad_y) / lbox.scale;

            // 裁剪到图像边界内。
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(image.width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(image.height)));

            candidates.push_back(RawDetection{x1, y1, x2, y2, max_score, best_class});
        }

        if (candidates.empty()) {
            return 0;
        }

        // 6. NMS。
        const std::vector<int> keep = nms_greedy(candidates, nms_threshold);

        // 7. 填充全部输出；调用层通过线程局部 vector 暴露结果，不再截断。
        if (keep.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return AI_ERR_RUNTIME;
        }
        const int32_t result_count = static_cast<int32_t>(keep.size());
        output->resize(keep.size());
        for (int32_t i = 0; i < result_count; ++i) {
            const auto& det = candidates[keep[i]];
            AIDetectBox& box = (*output)[static_cast<size_t>(i)];
            box.x1 = det.x1;
            box.y1 = det.y1;
            box.x2 = det.x2;
            box.y2 = det.y2;
            box.score = det.score;
            box.class_id = det.class_id;
            const std::string label = det.class_id >= 0 && static_cast<size_t>(det.class_id) < labels.size()
                ? labels[det.class_id] : std::string{};
            std::strncpy(box.label, label.c_str(), AIENGINE_MAX_LABEL - 1);
            box.label[AIENGINE_MAX_LABEL - 1] = 0;
        }

        return result_count;
}
} // namespace ai
