#include "ai_engine.h"
#include "compact_result_test_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string acp_from_wide(const std::wstring& value) {
#if defined(_WIN32)
    const int size = WideCharToMultiByte(CP_ACP, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    assert(size > 0);
    std::string output(static_cast<size_t>(size), '\0');
    assert(WideCharToMultiByte(CP_ACP, 0, value.data(), static_cast<int>(value.size()), output.data(), size, nullptr, nullptr) == size);
    return output;
#else
    return std::filesystem::path(value).u8string();
#endif
}

void assert_match(const CVMatchResult& result) {
    assert(result.x == 519);
    assert(result.y == 24);
    assert(result.w == 27);
    assert(result.h == 21);
    assert(std::fabs(result.sim - 1.0f) < 1e-6f);
}

} // namespace

int main() {
    const std::filesystem::path root = std::filesystem::absolute(
        std::filesystem::path(L"tests") / L"fixtures" / L"cv" / L"acp");
    const std::filesystem::path image_path = root / std::filesystem::u8path(u8"CV_大图.bmp");
    const std::filesystem::path template_dir = root / std::filesystem::u8path(u8"templates");
    const std::vector<uint8_t> image = read_file(image_path);

    const std::string directory_acp = acp_from_wide(template_dir.wstring());
    const std::string template_acp = acp_from_wide(L"CV_小图1.bmp");
#if defined(_WIN32)
    assert(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, template_acp.data(), static_cast<int>(template_acp.size()), nullptr, 0) > 0);
#endif

    int32_t handle = 0;
    const int32_t create_status = CV_Create(&handle);
    std::fprintf(stderr, "create=%d handle=%d\n", create_status, handle);
    assert(create_status == AI_OK && handle > 0);
    const int32_t load_status = CV_LoadTemplateDir(handle, directory_acp.c_str(), 0);
    const char* load_error = AI_GetLastError();
    std::fprintf(stderr, "load=%d error=%s\n", load_status, load_error);
    assert(load_status == 1);

    for (float threshold : {0.5f, 0.9f, 0.95f, 1.0f}) {
        CVMatchResult result{};
        const int32_t one_status = CV_FindOne(handle, template_acp.c_str(), image.data(), static_cast<int32_t>(image.size()), threshold, 0, &result);
        std::fprintf(stderr, "threshold=%.2f one=%d x=%d y=%d w=%d h=%d score=%.9f\n", threshold, one_status, result.x, result.y, result.w, result.h, result.sim);
        assert(one_status == 1);
        assert_match(result);

        result = CVMatchResult{};
        const int32_t transparent_status = CV_FindTransparentOne(handle, template_acp.c_str(), image.data(), static_cast<int32_t>(image.size()), threshold, 0, "FF00FF", &result);
        std::fprintf(stderr, "threshold=%.2f transparent=%d x=%d y=%d w=%d h=%d score=%.9f\n", threshold, transparent_status, result.x, result.y, result.w, result.h, result.sim);
        assert(transparent_status == 1);
        assert_match(result);

        const char* multi = CV_FindMultiText(handle, template_acp.c_str(), image.data(), static_cast<int32_t>(image.size()), "", threshold, 0);
        std::fprintf(stderr, "multi=%s\n", multi == nullptr ? "<null>" : multi);
        assert(multi != nullptr);
        assert(compact_test::contains(multi, 0, 519, 24));

        const char* transparent = CV_FindTransparentMultiText(handle, template_acp.c_str(), image.data(), static_cast<int32_t>(image.size()), "", threshold, "FF00FF");
        std::fprintf(stderr, "transparent_multi=%s\n", transparent == nullptr ? "<null>" : transparent);
        assert(transparent != nullptr);
        assert(compact_test::contains(transparent, 0, 519, 24));
    }

    CVMatchResult missing{};
    assert(CV_FindOne(handle, "missing.bmp", image.data(), static_cast<int32_t>(image.size()), 0.5f, 0, &missing) == AI_ERR_INVALID_ARGUMENT);
    const char* error = AI_GetLastError();
    assert(std::string(error).find("template not found") != std::string::npos);
    assert(CV_Release(handle) == AI_OK);
    return 0;
}
