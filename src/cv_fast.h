#pragma once

#include "ai_engine.h"

#include <stdint.h>
#include <string>

namespace ai {

// 将图像/ROI 像素转为 GRAY8，成功时返回像素数。
int32_t cv_to_gray(const AIImage& image, const AIRect* roi, uint8_t* output, int32_t output_stride);

// 使用固定灰度阈值，将图像/ROI 像素转为二值掩码。
int32_t cv_threshold(const AIImage& image, const AIRect* roi, int32_t threshold, uint8_t* output, int32_t output_stride);

// 提取细化后的前景中心线路径，并序列化为 JSON。
int32_t cv_extract_trace_json(const AIImage& image, const AIRect* roi, int32_t threshold, bool invert, int32_t max_points, std::string* output);

// 计算 ROI 内 B/G/R 均值以及灰度最小/最大值。
int32_t cv_mean_color(const AIImage& image, const AIRect* roi, AIColorStats* output);

// 查找与 target_bgr 在各通道容差范围内的像素。
int32_t cv_find_color(const AIImage& image, const AIRect* roi, uint32_t target_bgr, int32_t tolerance, AIColorFindResult* output);

// 查找最佳等尺寸模板匹配。
int32_t cv_find_image(const AIImage& image, const AIImage& templ, float min_score, AIImageMatch* output);

// 在模板数组中查找多个模板匹配。
int32_t cv_find_images(const AIImage& image, const AIImage* templates, int32_t template_count, float min_score, AIImageMatch* output, int32_t max_output);
int32_t cv_find_images_with_tolerance(const AIImage& image, const AIImage* templates, int32_t template_count, int32_t channel_tolerance, float min_score, AIImageMatch* output, int32_t max_output);

// 查找最佳模板匹配，同时忽略模板中的透明像素。
int32_t cv_find_transparent_image(const AIImage& image, const AIImage& templ, int32_t alpha_threshold, float min_score, AIImageMatch* output);

// 在模板数组中查找多个透明模板匹配。
int32_t cv_find_transparent_images(const AIImage& image, const AIImage* templates, int32_t template_count, int32_t alpha_threshold, float min_score, AIImageMatch* output, int32_t max_output);
int32_t cv_find_transparent_images_with_tolerance(const AIImage& image, const AIImage* templates, int32_t template_count, int32_t alpha_threshold, int32_t channel_tolerance, float min_score, AIImageMatch* output, int32_t max_output);

} // namespace ai
