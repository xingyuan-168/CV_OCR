#include "ai_engine.h"

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
    std::vector<uint8_t> result(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(result.data()), size);
    return input ? result : std::vector<uint8_t>{};
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
    const int output_size = WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0) return {};
    std::string output(static_cast<size_t>(output_size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8, 0, wide.data(), wide_size, output.data(), output_size,
            nullptr, nullptr) != output_size) {
        return {};
    }
    return output;
#else
    return input;
#endif
}

size_t json_line_count(const std::string& json) {
    size_t count = 0;
    size_t position = 0;
    while ((position = json.find("\"text\":\"", position)) != std::string::npos) {
        ++count;
        position += 8;
    }
    return count;
}

void require_contains(
    const std::string& text,
    const std::string& expected,
    const std::filesystem::path& path) {
    if (text.find(expected) == std::string::npos) {
        std::cerr << path.u8string() << " is missing expected text: " << expected
                  << "\nactual: " << text << "\n";
        std::abort();
    }
}

struct TestCase {
    const wchar_t* path;
    int32_t width;
    int32_t height;
    size_t minimum_lines;
    std::vector<std::string> required;
    std::string exact;
};

void verify_case(const TestCase& test_case) {
    const std::filesystem::path path(test_case.path);
    const std::vector<uint8_t> image = read_bytes(path);
    assert(!image.empty());
    std::string stable_text;
    std::string stable_json;
    for (int iteration = 0; iteration < 50; ++iteration) {
        const std::string text = acp_to_utf8(OCR_Recognize(
            image.data(),
            static_cast<int32_t>(image.size()),
            AI_OCR_OUTPUT_TEXT,
            0.5f,
            nullptr));
        const std::string json = acp_to_utf8(OCR_Recognize(
            image.data(),
            static_cast<int32_t>(image.size()),
            AI_OCR_OUTPUT_JSON,
            0.5f,
            nullptr));
        if (text.empty() || json.empty()) {
            std::cerr << path.u8string() << " returned an empty result: "
                      << AI_GetLastError() << "\n";
            std::abort();
        }
        for (const std::string& expected : test_case.required) {
            require_contains(text, expected, path);
        }
        if (!test_case.exact.empty() && text != test_case.exact) {
            std::cerr << path.u8string() << " exact text mismatch\nexpected: "
                      << test_case.exact << "\nactual: " << text << "\n";
            std::abort();
        }
        if (json_line_count(json) < test_case.minimum_lines) {
            std::cerr << path.u8string() << " returned too few text lines\n";
            std::abort();
        }
        const std::string whole_frame =
            "\"box\":{\"x\":0,\"y\":0,\"w\":" +
            std::to_string(test_case.width) + ",\"h\":" +
            std::to_string(test_case.height) + "}";
        if (json.find(whole_frame) != std::string::npos) {
            std::cerr << path.u8string()
                      << " incorrectly returned a whole-frame OCR line\n";
            std::abort();
        }
        if (iteration == 0) {
            stable_text = text;
            stable_json = json;
        } else if (text != stable_text || json != stable_json) {
            std::cerr << path.u8string()
                      << " produced an unstable result at iteration "
                      << iteration + 1 << "\n";
            std::abort();
        }
    }
    std::cout << path.u8string() << " lines=" << json_line_count(stable_json)
              << " text=" << stable_text << "\n";
}

} // namespace

int main() {
    const AIOcrRuntimeOptions options{0, 0, 4, 0.0f, 0.0f, 0.0f};
    if (OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 1, &options) != AI_OK) {
        std::cerr << "OCR load failed: " << AI_GetLastError() << "\n";
        return 2;
    }

    const std::vector<TestCase> cases{
        {L"tests\\fixtures\\ocr\\small_multiline\\1.bmp", 165, 173, 8,
         {u8"三界问情", u8"进阶坐骑", u8"新的力量"}, {}},
        {L"tests\\fixtures\\ocr\\small_multiline\\2.bmp", 131, 39, 2,
         {u8"长安酒店二楼", u8"[37,31]"}, u8"长安酒店二楼[37,31]"},
        {L"tests\\fixtures\\ocr\\small_multiline\\3.bmp", 151, 135, 6,
         {u8"材料二", u8"材料一"}, {}},
        {L"tests\\fixtures\\ocr\\small_multiline\\4.bmp", 144, 73, 4,
         {u8"法宝合成", u8"金钱镖"}, {}},
        {L"tests\\fixtures\\ocr\\small_multiline\\5.bmp", 167, 78, 4,
         {u8"法宝合成", u8"金钱镖"}, {}}
    };
    for (const TestCase& test_case : cases) verify_case(test_case);

    assert(OCR_Release() == AI_OK);
    AI_ShutdownWorker();
    return 0;
}
