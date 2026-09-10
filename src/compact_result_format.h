#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

struct CompactPointResult {
    int32_t id = 0;
    int32_t x = 0;
    int32_t y = 0;
};

// 易语言多目标接口的稳定文本格式：ID,x,y|ID,x,y。
// 空集合返回空文本；不添加空格、首尾分隔符或 JSON 标记。
inline std::string format_compact_points(const std::vector<CompactPointResult>& results) {
    std::string output;
    output.reserve(results.size() * 24);
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) output.push_back('|');
        output += std::to_string(results[i].id);
        output.push_back(',');
        output += std::to_string(results[i].x);
        output.push_back(',');
        output += std::to_string(results[i].y);
    }
    return output;
}

} // namespace ai
