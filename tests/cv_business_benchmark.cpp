#include "ai_engine.h"
#include "compact_result_test_utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
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
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

double percentile(std::vector<double> values, double ratio) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
        static_cast<size_t>(ratio * static_cast<double>(values.size() - 1)));
    return values[index];
}

template <typename Function>
std::vector<double> measure(int iterations, Function&& function) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        function();
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        samples.push_back(elapsed);
    }
    return samples;
}

void print_stats(const char* name, const std::vector<double>& samples) {
    double total = 0.0;
    for (double value : samples) total += value;
    std::cout << std::fixed << std::setprecision(3)
              << name << " avg=" << total / samples.size()
              << " p50=" << percentile(samples, 0.50)
              << " p95=" << percentile(samples, 0.95)
              << " min=" << *std::min_element(samples.begin(), samples.end())
              << " max=" << *std::max_element(samples.begin(), samples.end()) << " ms\n";
}

std::filesystem::path executable_directory() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size()) {
        path.resize(length);
        return std::filesystem::path(path).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

void verify_exe_relative_template_directory(const std::filesystem::path& template_dir) {
    const std::filesystem::path exe_dir = executable_directory();
    std::wstring relative_name = L"cv relative \u6a21\u677f";
#if defined(_WIN32)
    relative_name += L" " + std::to_wstring(GetCurrentProcessId());
#else
    relative_name += L" " + std::to_wstring(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
    const std::filesystem::path relative_dir = std::filesystem::path(relative_name);
    const std::filesystem::path target_dir = exe_dir / relative_dir;
    std::error_code error;
    std::filesystem::remove_all(target_dir, error);
    error.clear();
    std::filesystem::create_directories(target_dir, error);
    assert(!error);

    for (const auto& entry : std::filesystem::directory_iterator(template_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::filesystem::path extension = entry.path().extension();
        if (extension != L".bmp" && extension != L".BMP") continue;
        std::filesystem::copy_file(
            entry.path(), target_dir / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing, error);
        assert(!error);
    }

    int32_t handle = 0;
    assert(CV_Create(&handle) == AI_OK && handle > 0);
    const std::filesystem::path old_working_directory = std::filesystem::current_path();
    std::filesystem::current_path(std::filesystem::temp_directory_path());
    const std::string relative_utf8 = relative_dir.u8string();
    const int32_t loaded = CV_LoadTemplateDir(handle, relative_utf8.c_str(), 0);
    std::filesystem::current_path(old_working_directory);
    assert(loaded >= 2);
    assert(CV_Release(handle) == AI_OK);
    std::filesystem::remove_all(target_dir, error);
}

void verify_transparent_business_assets(const std::filesystem::path& assets) {
    const std::vector<uint8_t> big = read_bytes(assets / L"CV_大图.bmp");
    const std::filesystem::path template_dir = assets / L"templates";
    assert(!big.empty() && std::filesystem::is_directory(template_dir));

    int32_t handle = 0;
    assert(CV_Create(&handle) == AI_OK && handle > 0);
    const std::string template_dir_narrow = template_dir.string();
    assert(CV_LoadTemplateDir(handle, template_dir_narrow.c_str(), 0) >= 2);

    for (const char* color_bias : {"", "050505"}) {
        for (const float threshold : {0.3f, 0.5f, 0.9f, 1.0f}) {
            const char* value = CV_FindTransparentMultiText(
                handle, u8"xyq.bmp|梦.bmp", big.data(), static_cast<int32_t>(big.size()),
                color_bias, threshold, "FF00FF");
            assert(value != nullptr && value[0] != '\0');
            const std::string text(value);
            std::vector<compact_test::Point> points;
            assert(compact_test::parse(text, &points));
            assert(compact_test::contains(text, 0, 308, 99));
            assert(compact_test::contains(text, 1, 619, 135));
        }
    }

    CVMatchResult xyq{};
    CVMatchResult dream{};
    assert(CV_FindTransparentOne(handle, u8"xyq.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.9f, 0, "FF00FF", &xyq) == 1);
    assert(CV_FindTransparentOne(handle, u8"梦.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.9f, 0, "FF00FF", &dream) == 1);
    assert(xyq.x == 308 && xyq.y == 99 && std::abs(xyq.sim - 1.0f) <= 1e-6f);
    assert(dream.x == 619 && dream.y == 135 && std::abs(dream.sim - 1.0f) <= 1e-6f);

    std::vector<std::string> thread_json(4);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < thread_json.size(); ++i) {
        threads.emplace_back([&, i]() {
            const char* value = CV_FindTransparentMultiText(
                handle, u8"xyq.bmp|梦.bmp", big.data(), static_cast<int32_t>(big.size()),
                "050505", 0.3f, "FF00FF");
            thread_json[i] = value == nullptr ? "" : value;
        });
    }
    for (auto& thread : threads) thread.join();
    for (const std::string& text : thread_json) {
        assert(compact_test::contains(text, 0, 308, 99));
        assert(compact_test::contains(text, 1, 619, 135));
    }
    assert(CV_Release(handle) == AI_OK);
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path assets = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path(L"tests\\fixtures\\cv\\base");
    const int iterations = argc > 2 ? std::max(100, std::atoi(argv[2])) : 1000;
    const bool enforce = argc > 3 && std::string(argv[3]) == "--enforce";
    const std::string target = argc > 4 ? argv[4] : "all";
    if (target != "all" && target != "single" && target != "multi_json" && target != "multi_text" && target != "multi_results") {
        std::cerr << "unknown benchmark target\n";
        return 2;
    }
    verify_exe_relative_template_directory(assets);
    verify_transparent_business_assets(assets.parent_path() / L"transparent");
    const std::vector<uint8_t> big = read_bytes(assets / L"大图1.bmp");
    if (big.empty()) {
        std::cerr << "CV business BMP is missing\n";
        return 2;
    }

    int32_t cv_handle = 0;
    assert(CV_Create(&cv_handle) == AI_OK && cv_handle > 0);
    const std::string assets_narrow = assets.string();
    assert(CV_LoadTemplateDir(cv_handle, assets_narrow.c_str(), 0) >= 2);

    CVMatchResult one{};
    assert(CV_FindOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()), 0.4f, 0, &one) == 1);
    assert(one.x == 364 && one.y == 28);
    CVMatchResult transparent_one{};
    assert(CV_FindTransparentOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, "FF00FF", &transparent_one) == 1);
    assert(transparent_one.x == 364 && transparent_one.y == 28);

    const std::filesystem::path threshold_assets = assets.parent_path() / L"threshold";
    const std::vector<uint8_t> threshold_big = read_bytes(threshold_assets / L"CV_大图.bmp");
    assert(!threshold_big.empty());
    int32_t threshold_handle = 0;
    assert(CV_Create(&threshold_handle) == AI_OK && threshold_handle > 0);
    const std::string threshold_assets_narrow = threshold_assets.string();
    assert(CV_LoadTemplateDir(threshold_handle, threshold_assets_narrow.c_str(), 0) >= 1);
    for (float threshold : {0.5f, 0.8f, 0.9f, 0.95f, 1.0f}) {
        CVMatchResult exact{};
        assert(CV_FindOne(threshold_handle, u8"电.bmp", threshold_big.data(),
            static_cast<int32_t>(threshold_big.size()), threshold, 0, &exact) == 1);
        assert(exact.x == 502 && exact.y == 22);
        assert(std::abs(exact.sim - 1.0f) <= 1e-6f);
    }
    for (const char* transparent_rgb : {"FF00FF", "#FF00FF", "0xFF00FF"}) {
        CVMatchResult transparent{};
        assert(CV_FindTransparentOne(threshold_handle, u8"电.bmp", threshold_big.data(),
            static_cast<int32_t>(threshold_big.size()), 0.95f, 0, transparent_rgb, &transparent) == 1);
        assert(transparent.x == 502 && transparent.y == 22);
    }
    for (const char* invalid_rgb : {"", "FFFFF", "GG00FF", "0x1000000"}) {
        CVMatchResult invalid{};
        assert(CV_FindTransparentOne(threshold_handle, u8"电.bmp", threshold_big.data(),
            static_cast<int32_t>(threshold_big.size()), 0.5f, 0, invalid_rgb, &invalid) == AI_ERR_INVALID_ARGUMENT);
    }
    CVMatchResult absent{};
    assert(CV_FindOne(threshold_handle, "absent.bmp", threshold_big.data(),
        static_cast<int32_t>(threshold_big.size()), 1.0f, 0, &absent) == 0);
    const std::string gap_multi = CV_FindMultiText(
        threshold_handle, u8"absent.bmp|电.bmp", threshold_big.data(),
        static_cast<int32_t>(threshold_big.size()), "", 1.0f, 0);
    std::vector<compact_test::Point> gap_points;
    assert(compact_test::parse(gap_multi, &gap_points) && gap_points.size() == 1);
    assert(gap_points[0].id == 1 && gap_points[0].x == 502 && gap_points[0].y == 22);
    const std::string transparent_gap_multi = CV_FindTransparentMultiText(
        threshold_handle, u8"absent.bmp|电.bmp", threshold_big.data(),
        static_cast<int32_t>(threshold_big.size()), "", 1.0f, "FF00FF");
    gap_points.clear();
    assert(compact_test::parse(transparent_gap_multi, &gap_points) && gap_points.size() == 1);
    assert(gap_points[0].id == 1 && gap_points[0].x == 502 && gap_points[0].y == 22);
    CVMatchResult threshold_perf{};
    for (int i = 0; i < 50; ++i) {
        assert(CV_FindOne(threshold_handle, u8"电.bmp", threshold_big.data(),
            static_cast<int32_t>(threshold_big.size()), 0.95f, 0, &threshold_perf) == 1);
    }
    std::vector<double> threshold_single;
    if (target == "all" || target == "single") {
        threshold_single = measure(iterations, [&] {
            assert(CV_FindOne(threshold_handle, u8"电.bmp", threshold_big.data(),
                static_cast<int32_t>(threshold_big.size()), 0.95f, 0, &threshold_perf) == 1);
        });
        print_stats("threshold_single", threshold_single);
    }
    assert(CV_Release(threshold_handle) == AI_OK);

    auto verify_multi = [&] {
        const char* value = CV_FindMultiText(
            cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
            "", 0.4f, 0);
        assert(value != nullptr && value[0] != '\0');
        const std::string text(value);
        std::vector<compact_test::Point> points;
        assert(compact_test::parse(text, &points));
        assert(compact_test::contains(text, 0, 364, 28));
        assert(compact_test::contains(text, 1, 302, 18));
    };
    verify_multi();
    const char* transparent_text = CV_FindTransparentMultiText(
        cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, "FF00FF");
    assert(transparent_text != nullptr && transparent_text[0] != '\0');
    assert(compact_test::contains(transparent_text, 0, 364, 28));
    assert(compact_test::contains(transparent_text, 1, 302, 18));

    std::vector<uint8_t> blank_bmp = big;
    assert(blank_bmp.size() > 54);
    std::fill(blank_bmp.begin() + 54, blank_bmp.end(), 0);
    const char* transparent_no_match = CV_FindTransparentMultiText(
        cv_handle, u8"电.bmp|游.bmp", blank_bmp.data(), static_cast<int32_t>(blank_bmp.size()),
        "", 0.4f, "FF00FF");
    assert(transparent_no_match != nullptr && transparent_no_match[0] == '\0');
    assert(AI_GetLastError()[0] == '\0');

    const char* transparent_bad_color = CV_FindTransparentMultiText(
        cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, "not-rgb");
    assert(transparent_bad_color != nullptr && transparent_bad_color[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');

    const uint8_t invalid_bmp[] = {1, 2, 3, 4};
    const char* transparent_bad_bmp = CV_FindTransparentMultiText(
        cv_handle, u8"电.bmp|游.bmp", invalid_bmp, static_cast<int32_t>(sizeof(invalid_bmp)),
        "", 0.4f, "FF00FF");
    assert(transparent_bad_bmp != nullptr && transparent_bad_bmp[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');

    const char* transparent_missing_template = CV_FindTransparentMultiText(
        cv_handle, "missing.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, "FF00FF");
    assert(transparent_missing_template != nullptr && transparent_missing_template[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');

    const char* transparent_bad_handle = CV_FindTransparentMultiText(
        0, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, "FF00FF");
    assert(transparent_bad_handle != nullptr && transparent_bad_handle[0] == '\0');
    assert(AI_GetLastError()[0] != '\0');
    const char* text = CV_FindMultiText(
        cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, 0);
    assert(text != nullptr);
    assert(compact_test::contains(text, 0, 364, 28));
    assert(compact_test::contains(text, 1, 302, 18));

    for (int i = 0; i < 50; ++i) {
        assert(CV_FindOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()), 0.4f, 0, &one) == 1);
        assert(CV_FindTransparentOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
            0.4f, 0, "FF00FF", &transparent_one) == 1);
        verify_multi();
    }
    std::vector<std::string> thread_json(4);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < thread_json.size(); ++i) {
        threads.emplace_back([&, i]() {
            const char* value = CV_FindMultiText(
                cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
                "", 0.4f, 0);
            thread_json[i] = value == nullptr ? "" : value;
        });
    }
    for (auto& thread : threads) thread.join();
    for (const std::string& result_text : thread_json) {
        assert(compact_test::contains(result_text, 0, 364, 28));
        assert(compact_test::contains(result_text, 1, 302, 18));
    }
    std::vector<double> single;
    std::vector<double> transparent_single;
    std::vector<double> multi_text;
    std::vector<double> transparent_multi_text;
    if (target == "all" || target == "single") {
        single = measure(iterations, [&] {
            assert(CV_FindOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()), 0.4f, 0, &one) == 1);
        });
        print_stats("single", single);
        transparent_single = measure(iterations, [&] {
            assert(CV_FindTransparentOne(cv_handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
                0.4f, 0, "FF00FF", &transparent_one) == 1);
        });
        print_stats("transparent_single", transparent_single);
    }
    if (target == "all" || target == "multi_json" || target == "multi_text" || target == "multi_results") {
        multi_text = measure(iterations, [&] {
            const char* value = CV_FindMultiText(
                cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
                "", 0.4f, 0);
            assert(value != nullptr && value[0] != '\0');
        });
        print_stats("multi_text", multi_text);
        transparent_multi_text = measure(iterations, [&] {
            const char* value = CV_FindTransparentMultiText(
                cv_handle, u8"电.bmp|游.bmp", big.data(), static_cast<int32_t>(big.size()),
                "", 0.4f, "FF00FF");
            assert(value != nullptr && value[0] != '\0');
        });
        print_stats("transparent_multi_text", transparent_multi_text);
    }

    assert(CV_Release(cv_handle) == AI_OK);
    const bool gate_failed = (!single.empty() && percentile(single, 0.95) > 5.0) ||
        (!transparent_single.empty() && percentile(transparent_single, 0.95) > 5.0) ||
        (!multi_text.empty() && percentile(multi_text, 0.95) > 5.0) ||
        (!transparent_multi_text.empty() && percentile(transparent_multi_text, 0.95) > 5.0) ||
        (!threshold_single.empty() && percentile(threshold_single, 0.95) > 5.0);
    if (enforce && gate_failed) {
        std::cerr << "CV P95 performance gate failed\n";
        return 3;
    }
    return 0;
}
