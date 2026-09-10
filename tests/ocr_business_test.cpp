#include "ai_engine.h"
#include "compact_result_test_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
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
    return bytes;
}

std::string read_utf8(const std::filesystem::path& path) {
    const std::vector<uint8_t> bytes = read_bytes(path);
    std::string value(bytes.begin(), bytes.end());
    if (value.size() >= 3 && static_cast<uint8_t>(value[0]) == 0xEF &&
        static_cast<uint8_t>(value[1]) == 0xBB && static_cast<uint8_t>(value[2]) == 0xBF) {
        value.erase(0, 3);
    }
    return value;
}

std::string without_newlines(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
    value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
    return value;
}

std::string convert_code_page(const std::string& input, unsigned int from, unsigned int to) {
#if defined(_WIN32)
    if (input.empty()) return {};
    const DWORD decode_flags = from == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0;
    const DWORD encode_flags = to == CP_UTF8 ? WC_ERR_INVALID_CHARS : 0;
    const int wide_size = MultiByteToWideChar(from, decode_flags, input.data(),
        static_cast<int>(input.size()), nullptr, 0);
    assert(wide_size > 0);
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    assert(MultiByteToWideChar(from, decode_flags, input.data(),
        static_cast<int>(input.size()), wide.data(), wide_size) == wide_size);
    const int output_size = WideCharToMultiByte(to, encode_flags, wide.data(), wide_size,
        nullptr, 0, nullptr, nullptr);
    assert(output_size > 0);
    std::string output(static_cast<size_t>(output_size), '\0');
    assert(WideCharToMultiByte(to, encode_flags, wide.data(), wide_size,
        output.data(), output_size, nullptr, nullptr) == output_size);
    return output;
#else
    (void)from;
    (void)to;
    return input;
#endif
}

std::string utf8_to_acp(const std::string& input) {
#if defined(_WIN32)
    return convert_code_page(input, CP_UTF8, CP_ACP);
#else
    return input;
#endif
}

std::string acp_to_utf8(const char* input) {
    if (input == nullptr) return {};
#if defined(_WIN32)
    return convert_code_page(input, CP_ACP, CP_UTF8);
#else
    return input;
#endif
}

bool decode_bmp(std::vector<uint8_t>* bytes, AIImage* image) {
    if (bytes == nullptr || image == nullptr || bytes->size() < 54 || (*bytes)[0] != 'B' || (*bytes)[1] != 'M') return false;
    const auto u16 = [&](size_t offset) {
        return static_cast<uint16_t>((*bytes)[offset] | ((*bytes)[offset + 1] << 8));
    };
    const auto u32 = [&](size_t offset) {
        return static_cast<uint32_t>((*bytes)[offset] | ((*bytes)[offset + 1] << 8) |
            ((*bytes)[offset + 2] << 16) | ((*bytes)[offset + 3] << 24));
    };
    const uint32_t pixel_offset = u32(10);
    const int32_t width = static_cast<int32_t>(u32(18));
    const int32_t raw_height = static_cast<int32_t>(u32(22));
    const uint16_t bpp = u16(28);
    if (width <= 0 || raw_height == 0 || u32(30) != 0 || (bpp != 24 && bpp != 32)) return false;
    const int32_t height = raw_height < 0 ? -raw_height : raw_height;
    const int32_t stride = ((width * (bpp / 8) + 3) / 4) * 4;
    if (static_cast<uint64_t>(pixel_offset) + static_cast<uint64_t>(stride) * height > bytes->size()) return false;
    image->data = bytes->data() + pixel_offset;
    image->width = width;
    image->height = height;
    image->stride = stride;
    image->format = bpp == 24 ? AI_IMAGE_BGR24 : AI_IMAGE_BGRA32;
    if (raw_height > 0) {
        image->data += static_cast<size_t>(height - 1) * stride;
        image->stride = -stride;
    }
    return true;
}

int occurrences(const std::string& value, const std::string& needle) {
    int count = 0;
    for (size_t offset = 0; (offset = value.find(needle, offset)) != std::string::npos; offset += needle.size()) ++count;
    return count;
}

void require_contains(const std::string& value, const std::string& needle, const char* operation) {
    if (value.find(needle) == std::string::npos) {
        std::cerr << operation << " missing expected UTF-8 text: " << needle << "\nactual: " << value << "\n";
        std::abort();
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::cerr << "stage=read_assets\n";
    const std::filesystem::path assets = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path(L"tests/fixtures/ocr/business");
    std::vector<uint8_t> bmp = read_bytes(assets / L"OCR_大图.bmp");
    const std::string ground_truth = read_utf8(assets / L"OCR_大图_ground_truth.txt");
    const std::string expected_text = without_newlines(ground_truth);
    std::cerr << "bmp_size=" << bmp.size() << " truth_size=" << expected_text.size() << "\n";
    assert(!bmp.empty() && !expected_text.empty());

    const AIOcrRuntimeOptions options{0, 0, 4, 0.0f, 0.0f, 0.0f};
    const int32_t load_status = OCR_LoadEmbeddedModelEx(AI_DEVICE_CPU, 1, &options);
    if (load_status < 0) {
        const char* error = AI_GetLastError();
        std::cerr << "OCR_LoadEmbeddedModelEx failed: " << error << "\n";
        return 2;
    }
    std::cerr << "stage=compat_text\n";

    const std::string compat_text = acp_to_utf8(OCR_Recognize(
        bmp.data(), static_cast<int32_t>(bmp.size()), AI_OCR_OUTPUT_TEXT, 0.0f, nullptr));
    if (compat_text != expected_text) {
        std::cerr << "OCR_Recognize TEXT mismatch\nexpected: " << expected_text << "\nactual:   " << compat_text << "\n";
        return 3;
    }

    std::cerr << "stage=compat_json\n";
    const std::string compat_json = acp_to_utf8(OCR_Recognize(
        bmp.data(), static_cast<int32_t>(bmp.size()), AI_OCR_OUTPUT_JSON, 0.0f, nullptr));
    assert(compat_json.rfind("{\"lines\":[", 0) == 0);
    assert(occurrences(compat_json, "\"text\":") == 3);
    require_contains(compat_json, u8"最新动向:", "OCR_Recognize JSON");
    require_contains(compat_json, u8"1)新服为爱追寻[逍遥】于7月17日中午12:00开通。", "OCR_Recognize JSON");
    require_contains(compat_json, u8"2)《梦幻西游》又有好消息!目前已开创玩家最高同时在线达390万的新纪录!", "OCR_Recognize JSON");

    std::cerr << "stage=find_one_text\n";
    const std::string title_utf8 = u8"最新动向";
    const std::string title_acp = utf8_to_acp(title_utf8);
    OCRTextResult text_result{};
    const int32_t find_one_status = OCR_FindOneText(bmp.data(), static_cast<int32_t>(bmp.size()), title_acp.c_str(), 0.0f, &text_result, nullptr);
    if (find_one_status != 1) {
        const char* error = AI_GetLastError();
        std::cerr << "OCR_FindOneText failed: " << find_one_status << " " << error << "\n";
        return 4;
    }
    std::cerr << "find_one_box=" << text_result.x << "," << text_result.y << ","
              << text_result.w << "," << text_result.h << " center=" << text_result.cx << "," << text_result.cy << "\n";
    assert(text_result.w > 0 && text_result.h > 0);
    assert(text_result.cx == text_result.x + text_result.w / 2);
    assert(text_result.cy == text_result.y + text_result.h / 2);
    assert(text_result.x != text_result.cx && text_result.y != text_result.cy);
    assert(text_result.x == 3 && text_result.y == 6);
    assert(text_result.w == 51 && text_result.h == 17);
    assert(text_result.cx == 28 && text_result.cy == 14);
    static_assert(sizeof(OCRTextResult) == 28, "OCRTextResult must remain numeric-only");

    // Legacy callers may still reserve the former larger result buffer. The DLL
    // must write exactly the current 28-byte ABI and leave the tail untouched.
    for (const size_t size : {size_t{29}, size_t{156}, size_t{540}}) {
        std::vector<uint8_t> buffer(size, 0xa5);
        auto* result = reinterpret_cast<OCRTextResult*>(buffer.data());
        assert(OCR_FindOneText(
                   bmp.data(), static_cast<int32_t>(bmp.size()), title_acp.c_str(), 0.0f, result, nullptr) == 1);
        assert(result->x == text_result.x && result->y == text_result.y);
        for (size_t i = sizeof(OCRTextResult); i < buffer.size(); ++i) assert(buffer[i] == 0xa5);
    }

    std::cerr << "stage=find_multi_text\n";
    const std::string targets_acp = utf8_to_acp(u8"最新动向|梦幻西游|390万");
    const std::string multi_text = acp_to_utf8(OCR_FindMultiText(
        bmp.data(), static_cast<int32_t>(bmp.size()), targets_acp.c_str(), 0.0f, nullptr));
    std::vector<compact_test::Point> multi_points;
    assert(compact_test::parse(multi_text, &multi_points));
    assert(multi_points.size() == 3);
    assert(compact_test::contains(multi_text, 0, text_result.cx, text_result.cy));
    for (int32_t target_index = 0; target_index < 3; ++target_index) {
        assert(std::any_of(multi_points.begin(), multi_points.end(), [&](const compact_test::Point& point) {
            return point.id == target_index;
        }));
    }
    const std::string gap_targets_acp = utf8_to_acp(u8"不存在|梦幻西游");
    const std::string gap_text = acp_to_utf8(OCR_FindMultiText(
        bmp.data(), static_cast<int32_t>(bmp.size()), gap_targets_acp.c_str(), 0.0f, nullptr));
    std::vector<compact_test::Point> gap_points;
    assert(compact_test::parse(gap_text, &gap_points) && !gap_points.empty());
    assert(std::all_of(gap_points.begin(), gap_points.end(), [](const compact_test::Point& point) {
        return point.id == 1;
    }));
    const std::string missing_target_acp = utf8_to_acp(u8"完全不存在的目标");
    assert(OCR_FindMultiText(
        bmp.data(), static_cast<int32_t>(bmp.size()), missing_target_acp.c_str(), 0.0f, nullptr)[0] == '\0');
    assert(AI_GetLastError()[0] == '\0');

    std::cerr << "stage=find_one_coord\n";
    OCRCoordResult coord{};
    assert(OCR_FindOneCoord(bmp.data(), static_cast<int32_t>(bmp.size()), title_acp.c_str(), 0.0f, &coord, nullptr) == 1);
    assert(coord.x == text_result.cx && coord.y == text_result.cy && coord.w == text_result.w && coord.h == text_result.h);

#if !defined(_M_IX86)
    std::cerr << "stage=standard_utf8\n";
    AIImage image{};
    assert(decode_bmp(&bmp, &image));
    const std::string standard_json = AI_OcrRecognize(&image, 0.0f);
    assert(standard_json.front() == '[' && standard_json.back() == ']');
    assert(occurrences(standard_json, "\"text\":") == 3);
    require_contains(standard_json, u8"最新动向:", "AI_OcrRecognize");
    const std::string standard_find = AI_OcrFindText(&image, u8"梦幻西游", 0.0f);
    assert(standard_find.front() == '[' && standard_find.back() == ']');
    require_contains(standard_find, u8"梦幻西游", "AI_OcrFindText");
#endif

    std::cerr << "stage=release\n";
    assert(OCR_Release() == AI_OK);
    std::cout << "OCR business ground truth passed: " << expected_text << "\n";
    return 0;
}
