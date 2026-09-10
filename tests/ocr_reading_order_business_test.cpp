#include "ai_engine.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0) return {};
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    return input ? bytes : std::vector<uint8_t>{};
}

std::string acp_to_utf8(const char* input) {
    if (input == nullptr || input[0] == '\0') return {};
#if defined(_WIN32)
    const int input_size = static_cast<int>(std::strlen(input));
    const int wide_size = MultiByteToWideChar(CP_ACP, 0, input, input_size, nullptr, 0);
    if (wide_size <= 0) return {};
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(CP_ACP, 0, input, input_size, wide.data(), wide_size) != wide_size) return {};
    const int output_size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0) return {};
    std::string output(static_cast<size_t>(output_size), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_size, output.data(), output_size, nullptr, nullptr) != output_size) return {};
    return output;
#else
    return input;
#endif
}

std::vector<std::string> json_line_texts(const std::string& json) {
    std::vector<std::string> lines;
    const std::string marker = "\"text\":\"";
    size_t position = 0;
    while ((position = json.find(marker, position)) != std::string::npos) {
        position += marker.size();
        std::string text;
        bool escaped = false;
        for (; position < json.size(); ++position) {
            const char ch = json[position];
            if (escaped) {
                switch (ch) {
                    case '"': text.push_back('"'); break;
                    case '\\': text.push_back('\\'); break;
                    case 'n': text.push_back('\n'); break;
                    case 'r': text.push_back('\r'); break;
                    case 't': text.push_back('\t'); break;
                    default: text.push_back(ch); break;
                }
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                ++position;
                break;
            } else {
                text.push_back(ch);
            }
        }
        lines.push_back(std::move(text));
    }
    return lines;
}

double percentile95_ms(std::vector<int64_t> values) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1;
    return static_cast<double>(values[std::min(index, values.size() - 1)]) / 1000.0;
}

double verify_case(
    const std::filesystem::path& path,
    const std::string& expected,
    size_t expected_lines,
    const std::vector<std::string>& ordered_box_markers) {
    const std::vector<uint8_t> image = read_bytes(path);
    assert(!image.empty());
    std::vector<int64_t> latencies;
    latencies.reserve(100);
    std::string first_json;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const std::string text = acp_to_utf8(OCR_Recognize(
            image.data(), static_cast<int32_t>(image.size()), AI_OCR_OUTPUT_TEXT, 0.0f, nullptr));
        latencies.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
        if (text != expected) {
            std::cerr << "TEXT mismatch at iteration " << iteration + 1 << "\nexpected: " << expected
                      << "\nactual:   " << text << "\n";
            std::abort();
        }

        const std::string json = acp_to_utf8(OCR_Recognize(
            image.data(), static_cast<int32_t>(image.size()), AI_OCR_OUTPUT_JSON, 0.0f, nullptr));
        if (iteration == 0) first_json = json;
        const std::vector<std::string> lines = json_line_texts(json);
        assert(lines.size() == expected_lines);
        std::string joined;
        for (const auto& line : lines) joined += line;
        assert(joined == expected);
        size_t marker_position = 0;
        for (const auto& marker : ordered_box_markers) {
            marker_position = json.find(marker, marker_position);
            assert(marker_position != std::string::npos);
            marker_position += marker.size();
        }
    }

    const double average_ms = static_cast<double>(
        std::accumulate(latencies.begin(), latencies.end(), int64_t{0})) / latencies.size() / 1000.0;
    const double p95_ms = percentile95_ms(latencies);
    std::cout << path.u8string() << " iterations=100 average_ms=" << average_ms
              << " p95_ms=" << p95_ms << "\n" << first_json << "\n" << std::flush;
    return p95_ms;
}

}  // namespace

int main() {
    const AIOcrRuntimeOptions options{0, 0, 4, 0.0f, 0.0f, 0.0f};
    assert(OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 1, &options) == AI_OK);

    verify_case(
        std::filesystem::path(L"tests\\fixtures\\ocr\\reading_order\\OCR测试.bmp"),
        u8"2)《梦幻西游》又有好消息!目前已开创玩家最高同时在线达390万的新纪录!",
        2,
        {"\"box\":{\"x\":0,\"y\":11", "\"box\":{\"x\":16,\"y\":7"});
    const double multi_p95_ms = verify_case(
        std::filesystem::path(L"tests\\fixtures\\ocr\\reading_order\\OCR测试2.bmp"),
        u8"最新动向:1)新服为爱追寻[逍遥】于7月17日中午12:00开通。2)《梦幻西游》又有好消息!目前已开创玩家最高同时在线达390万的新纪录!",
        3,
        {"\"box\":{\"x\":2,\"y\":5", "\"box\":{\"x\":1,\"y\":31", "\"box\":{\"x\":0,\"y\":58"});
    verify_case(
        std::filesystem::path(L"tests\\fixtures\\ocr\\business\\OCR_大图.bmp"),
        u8"最新动向:1)新服为爱追寻[逍遥】于7月17日中午12:00开通。2)《梦幻西游》又有好消息!目前已开创玩家最高同时在线达390万的新纪录!",
        3,
        {"\"box\":{\"x\":3,\"y\":6", "\"box\":{\"x\":2,\"y\":33", "\"box\":{\"x\":1,\"y\":57"});

    // Latency is hardware and provider dependent.  The v23 performance
    // benchmark owns the baseline-relative P50/P95 gates; this functional test
    // only verifies deterministic text, JSON lines, boxes, and reading order.
    assert(std::isfinite(multi_p95_ms) && multi_p95_ms > 0.0);
    assert(OCR_Release() == AI_OK);
    AI_ShutdownWorker();
    return 0;
}
