#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace compact_test {

struct Point {
    int32_t id = 0;
    int32_t x = 0;
    int32_t y = 0;
};

inline bool parse_int32(std::string_view text, int32_t* value) {
    if (value == nullptr || text.empty()) return false;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, *value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

inline bool parse(const std::string& text, std::vector<Point>* points) {
    if (points == nullptr) return false;
    points->clear();
    if (text.empty()) return true;

    size_t item_start = 0;
    while (item_start < text.size()) {
        const size_t item_end = text.find('|', item_start);
        const size_t end = item_end == std::string::npos ? text.size() : item_end;
        if (end == item_start) return false;
        const std::string_view item(text.data() + item_start, end - item_start);
        const size_t first_comma = item.find(',');
        const size_t second_comma = first_comma == std::string_view::npos
            ? std::string_view::npos
            : item.find(',', first_comma + 1);
        if (first_comma == std::string_view::npos ||
            second_comma == std::string_view::npos ||
            item.find(',', second_comma + 1) != std::string_view::npos) {
            return false;
        }
        Point point{};
        if (!parse_int32(item.substr(0, first_comma), &point.id) ||
            !parse_int32(item.substr(first_comma + 1, second_comma - first_comma - 1), &point.x) ||
            !parse_int32(item.substr(second_comma + 1), &point.y) || point.id < 0) {
            return false;
        }
        points->push_back(point);
        if (item_end == std::string::npos) break;
        item_start = item_end + 1;
        if (item_start == text.size()) return false;
    }
    return true;
}

inline bool contains(const std::string& text, int32_t id, int32_t x, int32_t y) {
    std::vector<Point> points;
    if (!parse(text, &points)) return false;
    for (const Point& point : points) {
        if (point.id == id && point.x == x && point.y == y) return true;
    }
    return false;
}

}  // namespace compact_test
