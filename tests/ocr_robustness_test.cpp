#include "ai_engine.h"
#include "compact_result_test_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    const int wide_size = MultiByteToWideChar(
        CP_ACP, 0, input, input_size, nullptr, 0);
    if (wide_size <= 0) return {};
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(
            CP_ACP, 0, input, input_size, wide.data(), wide_size) != wide_size) {
        return {};
    }
    const int utf8_size = WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    if (utf8_size <= 0) return {};
    std::string output(static_cast<size_t>(utf8_size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8, 0, wide.data(), wide_size, output.data(), utf8_size,
            nullptr, nullptr) != utf8_size) return {};
    return output;
#else
    return input;
#endif
}

std::string utf8_to_acp(const std::string& input) {
#if defined(_WIN32)
    const int wide_size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
        static_cast<int>(input.size()), nullptr, 0);
    assert(wide_size > 0);
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    assert(MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
        static_cast<int>(input.size()), wide.data(), wide_size) == wide_size);
    const int output_size = WideCharToMultiByte(
        CP_ACP, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    assert(output_size > 0);
    std::string output(static_cast<size_t>(output_size), '\0');
    assert(WideCharToMultiByte(
        CP_ACP, 0, wide.data(), wide_size, output.data(), output_size,
        nullptr, nullptr) == output_size);
    return output;
#else
    return input;
#endif
}

int json_box_value(const std::string& json, const char* key) {
    const size_t box = json.find("\"box\":{");
    assert(box != std::string::npos);
    const std::string marker = std::string("\"") + key + "\":";
    const size_t begin = json.find(marker, box);
    assert(begin != std::string::npos);
    return std::stoi(json.substr(begin + marker.size()));
}

void verify_origin_behavior(const std::vector<uint8_t>& image) {
    constexpr int32_t origin_x = -231;
    constexpr int32_t origin_y = 417;
    const std::string target = utf8_to_acp(u8"宫殿(一般)");
    const std::string missing = utf8_to_acp(u8"绝不可能命中");

    const std::string baseline_text = acp_to_utf8(OCR_Recognize(
        image.data(), static_cast<int32_t>(image.size()),
        AI_OCR_OUTPUT_TEXT, 0.5f, "FFFFFF-202020", 0, 0));
    const std::string shifted_text = acp_to_utf8(OCR_Recognize(
        image.data(), static_cast<int32_t>(image.size()),
        AI_OCR_OUTPUT_TEXT, 0.5f, "FFFFFF-202020", origin_x, origin_y));
    assert(shifted_text == baseline_text);

    const std::string baseline_json = acp_to_utf8(OCR_Recognize(
        image.data(), static_cast<int32_t>(image.size()),
        AI_OCR_OUTPUT_JSON, 0.5f, "FFFFFF-202020", 0, 0));
    const std::string shifted_json = acp_to_utf8(OCR_Recognize(
        image.data(), static_cast<int32_t>(image.size()),
        AI_OCR_OUTPUT_JSON, 0.5f, "FFFFFF-202020", origin_x, origin_y));
    assert(json_box_value(shifted_json, "x") == json_box_value(baseline_json, "x") + origin_x);
    assert(json_box_value(shifted_json, "y") == json_box_value(baseline_json, "y") + origin_y);
    assert(json_box_value(shifted_json, "w") == json_box_value(baseline_json, "w"));
    assert(json_box_value(shifted_json, "h") == json_box_value(baseline_json, "h"));

    OCRTextResult baseline_text_result{};
    OCRTextResult shifted_text_result{};
    assert(OCR_FindOneText(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        &baseline_text_result, "FFFFFF-202020", 0, 0) == 1);
    assert(OCR_FindOneText(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        &shifted_text_result, "FFFFFF-202020", origin_x, origin_y) == 1);
    assert(shifted_text_result.x == baseline_text_result.x + origin_x);
    assert(shifted_text_result.y == baseline_text_result.y + origin_y);
    assert(shifted_text_result.cx == baseline_text_result.cx + origin_x);
    assert(shifted_text_result.cy == baseline_text_result.cy + origin_y);
    assert(shifted_text_result.w == baseline_text_result.w);
    assert(shifted_text_result.h == baseline_text_result.h);

    OCRCoordResult baseline_coord{};
    OCRCoordResult shifted_coord{};
    assert(OCR_FindOneCoord(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        &baseline_coord, "FFFFFF-202020", 0, 0) == 1);
    assert(OCR_FindOneCoord(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        &shifted_coord, "FFFFFF-202020", origin_x, origin_y) == 1);
    assert(shifted_coord.x == baseline_coord.x + origin_x);
    assert(shifted_coord.y == baseline_coord.y + origin_y);
    assert(shifted_coord.w == baseline_coord.w && shifted_coord.h == baseline_coord.h);

    const std::string baseline_multi = acp_to_utf8(OCR_FindMultiText(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        "FFFFFF-202020", 0, 0));
    const std::string shifted_multi = acp_to_utf8(OCR_FindMultiText(
        image.data(), static_cast<int32_t>(image.size()), target.c_str(), 0.5f,
        "FFFFFF-202020", origin_x, origin_y));
    std::vector<compact_test::Point> baseline_points;
    std::vector<compact_test::Point> shifted_points;
    assert(compact_test::parse(baseline_multi, &baseline_points) && baseline_points.size() == 1);
    assert(compact_test::parse(shifted_multi, &shifted_points) && shifted_points.size() == 1);
    assert(shifted_points[0].id == baseline_points[0].id);
    assert(shifted_points[0].x == baseline_points[0].x + origin_x);
    assert(shifted_points[0].y == baseline_points[0].y + origin_y);
    assert(std::strcmp(OCR_FindMultiText(
        image.data(), static_cast<int32_t>(image.size()), missing.c_str(), 0.5f,
        "FFFFFF-202020", origin_x, origin_y), "") == 0);
    assert(AI_GetLastError()[0] == '\0');

    OCRTextResult missing_result{1, 2, 3, 4, 5, 6, 0.5f};
    assert(OCR_FindOneText(
        image.data(), static_cast<int32_t>(image.size()), missing.c_str(), 0.5f,
        &missing_result, "FFFFFF-202020", origin_x, origin_y) == 0);
    const OCRTextResult zero{};
    assert(std::memcmp(&missing_result, &zero, sizeof(zero)) == 0);
}

void verify_filter(
    const std::vector<uint8_t>& image,
    const char* filter,
    float min_confidence,
    int iterations) {
    const std::string expected = u8"副本宫殿(一般)";
    const std::string target = utf8_to_acp(u8"宫殿(一般)");
    std::string stable_json;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string text = acp_to_utf8(OCR_Recognize(
            image.data(), static_cast<int32_t>(image.size()),
            AI_OCR_OUTPUT_TEXT, min_confidence, filter));
        const std::string text_error = AI_GetLastError();
        const std::string json = acp_to_utf8(OCR_Recognize(
            image.data(), static_cast<int32_t>(image.size()),
            AI_OCR_OUTPUT_JSON, min_confidence, filter));
        if (text != expected || json.find(expected) == std::string::npos ||
            std::count(json.begin(), json.end(), '{') < 2) {
            std::cerr << "OCR robustness mismatch at iteration " << iteration + 1
                      << " confidence=" << min_confidence
                      << " filter=" << (filter == nullptr ? "auto" : filter)
                      << "\ntext=" << text << "\njson=" << json
                      << "\ntext_error=" << text_error
                      << "\njson_error=" << AI_GetLastError() << "\n";
            std::abort();
        }
        OCRTextResult match{};
        const int32_t find_status = OCR_FindOneText(
            image.data(), static_cast<int32_t>(image.size()), target.c_str(),
            min_confidence, &match, filter);
        if (find_status != 1) {
            std::cerr << "Find failed at iteration " << iteration + 1
                      << " confidence=" << min_confidence
                      << " filter=" << (filter == nullptr ? "auto" : filter)
                      << " status=" << find_status
                      << " error=" << AI_GetLastError()
                      << "\ntext=" << text << "\njson=" << json << "\n";
            std::abort();
        }
        if (match.w <= 0 || match.h <= 0 || match.x < 0 || match.y < 0 ||
            match.x + match.w > 150 || match.y + match.h > 28 ||
            match.cx != match.x + match.w / 2 ||
            match.cy != match.y + match.h / 2) {
            std::cerr << "Find geometry changed at iteration " << iteration + 1
                      << " confidence=" << min_confidence
                      << " filter=" << (filter == nullptr ? "auto" : filter)
                      << " box=" << match.x << ',' << match.y << ','
                      << match.w << ',' << match.h << " center="
                      << match.cx << ',' << match.cy << "\njson=" << json
                      << "\n";
            std::abort();
        }
        const int json_x = json_box_value(json, "x");
        const int json_y = json_box_value(json, "y");
        const int json_w = json_box_value(json, "w");
        const int json_h = json_box_value(json, "h");
        if (match.x < json_x || match.y < json_y ||
            match.x + match.w > json_x + json_w ||
            match.y + match.h > json_y + json_h) {
            std::cerr << "Find box escaped JSON line at iteration "
                      << iteration + 1 << " confidence=" << min_confidence
                      << " filter=" << (filter == nullptr ? "auto" : filter)
                      << "\njson=" << json << "\n";
            std::abort();
        }
        if (iteration == 0) stable_json = json;
        else if (json != stable_json) {
            std::cerr << "JSON drift at iteration " << iteration + 1
                      << " confidence=" << min_confidence
                      << " filter=" << (filter == nullptr ? "auto" : filter)
                      << "\nexpected=" << stable_json << "\nactual=" << json
                      << "\n";
            std::abort();
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ocr_robustness_test <fixture.bmp> [device] [iterations]\n";
        return 2;
    }
    const std::filesystem::path path =
        std::filesystem::u8path(acp_to_utf8(argv[1]));
    const std::vector<uint8_t> image = read_bytes(path);
    assert(!image.empty());
    const int32_t device = argc >= 3 ? std::atoi(argv[2]) : AI_DEVICE_CPU;
    const int iterations = argc >= 4 ? std::atoi(argv[3]) : 50;
    assert(iterations > 0);
    const AIOcrRuntimeOptions options{0, 0, 4, 0.0f, 0.0f, 0.0f};
    if (OCR_LoadEmbeddedModelEx(device, 1, &options) != AI_OK) {
        std::cerr << "OCR load failed: " << AI_GetLastError() << "\n";
        return 3;
    }
    for (const float confidence : {0.0f, 0.45f, 0.5f, 0.8f}) {
        verify_filter(image, "FFFFFF-202020", confidence, iterations);
        verify_filter(image, nullptr, confidence, iterations);
    }
    verify_origin_behavior(image);
    assert(OCR_Release() == AI_OK);
    AI_ShutdownWorker();
    return 0;
}
