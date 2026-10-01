#include "ai_engine.h"
#if defined(AIENGINE_CV_TEST_HOOKS)
#include "cv_test_hooks.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;

Bytes read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(input), {});
}

uint32_t u32(const Bytes& bytes, size_t offset) {
    uint32_t value = 0;
    assert(offset + sizeof(value) <= bytes.size());
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

struct Bmp {
    Bytes bytes;
    int width, height, stride, offset;
    bool bottom_up;
    explicit Bmp(Bytes data) : bytes(std::move(data)) {
        assert(bytes.size() >= 54 && bytes[0] == 'B' && bytes[1] == 'M');
        assert(bytes[28] == 24 && bytes[29] == 0);
        width = static_cast<int>(u32(bytes, 18));
        const int signed_height = static_cast<int>(u32(bytes, 22));
        assert(width > 0 && signed_height != 0 && signed_height != INT32_MIN);
        bottom_up = signed_height > 0;
        height = std::abs(signed_height);
        stride = (width * 3 + 3) & ~3;
        offset = static_cast<int>(u32(bytes, 10));
        assert(offset >= 54 && static_cast<size_t>(offset) +
            static_cast<size_t>(height) * stride <= bytes.size());
    }
    uint8_t* row(int y) {
        return bytes.data() + offset + (bottom_up ? height - 1 - y : y) * stride;
    }
};

struct Answer { int status; CVMatchResult match; };

void paste_visible(Bmp& frame, Bmp& templ, int x, int y) {
    for (int ty = 0; ty < templ.height; ++ty) {
        uint8_t* dst = frame.row(y + ty) + x * 3;
        const uint8_t* src = templ.row(ty);
        for (int tx = 0; tx < templ.width; ++tx) {
            const uint8_t* pixel = src + tx * 3;
            if (pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 255) continue;
            std::memcpy(dst + tx * 3, pixel, 3);
        }
    }
}

Answer find(int handle, const char* name, const Bmp& frame, float threshold = 0.9f,
            int mode = 0) {
    Answer answer{};
    answer.status = CV_FindTransparentOne(handle, name, frame.bytes.data(),
        static_cast<int>(frame.bytes.size()), threshold, mode, "FF00FF", &answer.match);
    assert(answer.status >= 0);
    assert(AI_GetLastError()[0] == '\0');
    return answer;
}

void compare(int handle, const char* name, const Bmp& frame, float threshold,
             int mode) {
#if defined(AIENGINE_CV_TEST_HOOKS)
    CVTest_Reference(1);
    const Answer expected = find(handle, name, frame, threshold, mode);
    CVTest_Reference(0);
    const Answer actual = find(handle, name, frame, threshold, mode);
    assert(actual.status == expected.status);
    if (actual.status == 1) {
        assert(actual.match.x == expected.match.x &&
            actual.match.y == expected.match.y &&
            actual.match.w == expected.match.w &&
            actual.match.h == expected.match.h &&
            actual.match.template_index == expected.match.template_index &&
            actual.match.sim == expected.match.sim);
    }
    CVTest_ForceScalarMasked(1);
    const Answer scalar = find(handle, name, frame, threshold, mode);
    CVTest_ForceScalarMasked(0);
    assert(scalar.status == expected.status);
    if (scalar.status == 1) {
        assert(scalar.match.x == expected.match.x &&
            scalar.match.y == expected.match.y &&
            scalar.match.sim == expected.match.sim);
    }
#else
    (void)handle; (void)name; (void)frame; (void)threshold; (void)mode;
#endif
}

double benchmark(int handle, const char* label, const char* name,
                 const std::vector<Bmp>& frames, int iterations) {
    for (int i = 0; i < 64; ++i) (void)find(handle, name, frames[i % frames.size()]);
    std::vector<double> samples;
    samples.reserve(iterations);
    int hits = 0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const Answer result = find(handle, name, frames[i % frames.size()]);
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
        hits += result.status == 1;
    }
    std::sort(samples.begin(), samples.end());
    const double p95 = samples[static_cast<size_t>((samples.size() - 1) * 0.95)];
    std::cout << label << " samples=" << iterations << " hits=" << hits
        << " p50=" << samples[samples.size() / 2]
        << " p95=" << p95 << " max=" << samples.back() << " ms\n";
#if defined(AIENGINE_CV_TEST_HOOKS)
    CVTestTransparentProfile profile{};
    CVTest_TransparentProfile(&profile);
    std::cout << "  stages rough=" << profile.rough_ms
        << " prepare=" << profile.prepare_ms
        << " heap=" << profile.heap_ms
        << " exact=" << profile.exact_ms
        << " candidates=" << profile.candidates
        << " tile_refreshes=" << profile.tile_refreshes << '\n';
#endif
    return p95;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: cv_real_transparent_benchmark INPUT_DIR [ITERATIONS [--enforce] | --stress THREADS SECONDS]\n";
        return 2;
    }
    const std::filesystem::path root = std::filesystem::absolute(argv[1]);
    const bool stress = argc > 2 && std::string(argv[2]) == "--stress";
    const int iterations = argc > 2 && std::string(argv[2]) == "--profile" ? 100 :
        (argc > 2 ? std::max(1000, std::atoi(argv[2])) : 1000);
    const bool enforce = !stress && argc > 3 && std::string(argv[3]) == "--enforce";
    Bmp original(read(root / std::filesystem::u8path(u8"CV_大图.bmp")));
    Bmp target(read(root / "xyq.bmp"));
    int handle = 0;
    assert(CV_Create(&handle) == AI_OK);
    assert(CV_LoadTemplateDir(handle, root.u8string().c_str(), 0) >= 2);

    const Answer hit = find(handle, "xyq.bmp", original);
    const Answer miss = find(handle, "CV_123.bmp", original);
    assert(hit.status == 1 && hit.match.x == 324 && hit.match.y == 105);
    assert(miss.status == 0);
    compare(handle, "xyq.bmp", original, 0.9f, 0);
    compare(handle, "CV_123.bmp", original, 0.9f, 0);
    for (float threshold : {0.0f, 0.3f, 0.899999f, 0.9f, 0.999999f, 1.0f}) {
        compare(handle, "xyq.bmp", original, threshold, 0);
        compare(handle, "CV_123.bmp", original, threshold, 0);
    }
    compare(handle, "xyq.bmp", original, 0.9f, 1);
    Bmp tied(original.bytes);
    for (int y = 0; y < tied.height; ++y) {
        std::fill(tied.row(y), tied.row(y) + tied.width * 3, uint8_t{0});
    }
    paste_visible(tied, target, 40, 40);
    paste_visible(tied, target, 180, 40);
    compare(handle, "xyq.bmp", tied, 0.9f, 0);
    const Answer first_tie = find(handle, "xyq.bmp", tied);
    assert(first_tie.status == 1 && first_tie.match.x == 40 && first_tie.match.y == 40);

    std::vector<Bmp> constant{original};
    std::vector<Bmp> changed;
    changed.reserve(32);
    for (int i = 0; i < 32; ++i) {
        Bmp frame(original.bytes);
        // Change background content without touching the known target.
        for (int y = 0; y < frame.height; ++y) {
            uint8_t* row = frame.row(y);
            for (int x = 0; x < frame.width; ++x) {
                if (x >= 324 && x < 324 + target.width &&
                    y >= 105 && y < 105 + target.height) continue;
                if (((x * 13 + y * 7 + i * 11) & 63) == 0)
                    row[x * 3 + (i % 3)] ^= static_cast<uint8_t>(i + 1);
            }
        }
        if (i % 2) {
            // Remove the old visible pixels and place the target elsewhere.
            for (int y = 0; y < target.height; ++y) {
                uint8_t* old_row = frame.row(105 + y) + 324 * 3;
                std::memcpy(old_row, frame.row(180 + y) + 600 * 3,
                    target.width * 3);
            }
            const int x = 30 + (i * 17) % 250;
            const int y = 30 + (i * 13) % 100;
            paste_visible(frame, target, x, y);
        }
        compare(handle, "xyq.bmp", frame, 0.9f, 0);
        compare(handle, "CV_123.bmp", frame, 0.9f, 0);
        changed.push_back(std::move(frame));
    }
    if (stress) {
        assert(argc >= 5);
        const int thread_count = std::atoi(argv[3]);
        const int seconds = std::atoi(argv[4]);
        assert(thread_count >= 1 && thread_count <= 64 && seconds >= 1);
        std::vector<Answer> hit_expected, miss_expected;
        for (const Bmp& frame : changed) {
            hit_expected.push_back(find(handle, "xyq.bmp", frame));
            miss_expected.push_back(find(handle, "CV_123.bmp", frame));
        }
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<uint64_t> calls{0};
        std::vector<std::vector<double>> samples(thread_count);
        std::vector<std::thread> workers;
        std::chrono::steady_clock::time_point deadline;
        for (int t = 0; t < thread_count; ++t) {
            workers.emplace_back([&, t] {
                samples[t].reserve(10000);
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                for (uint64_t i = 0; std::chrono::steady_clock::now() < deadline; ++i) {
                    const size_t index = static_cast<size_t>((i + t * 7) % changed.size());
                    const bool hit_case = (i & 1) == 0;
                    const auto start = std::chrono::steady_clock::now();
                    const Answer actual = find(handle, hit_case ? "xyq.bmp" : "CV_123.bmp", changed[index]);
                    const double elapsed = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    const Answer& expected = hit_case ? hit_expected[index] : miss_expected[index];
                    assert(actual.status == expected.status);
                    if (actual.status == 1) {
                        assert(actual.match.x == expected.match.x &&
                            actual.match.y == expected.match.y &&
                            actual.match.sim == expected.match.sim);
                    }
                    if (samples[t].size() < 10000) samples[t].push_back(elapsed);
                    else samples[t][i % 10000] = elapsed;
                    calls.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        while (ready.load(std::memory_order_acquire) != thread_count) std::this_thread::yield();
        const auto start = std::chrono::steady_clock::now();
        deadline = start + std::chrono::seconds(seconds);
        go.store(true, std::memory_order_release);
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
#if defined(AIENGINE_CV_TEST_HOOKS)
            const CVTestStats stats = CVTest_Stats();
            assert(stats.active <= stats.limit && stats.idle_bytes <= stats.budget);
#endif
        }
        for (auto& worker : workers) worker.join();
        std::vector<double> all;
        for (const auto& thread_samples : samples)
            all.insert(all.end(), thread_samples.begin(), thread_samples.end());
        std::sort(all.begin(), all.end());
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "transparent_stress threads=" << thread_count
            << " calls=" << calls.load() << " seconds=" << duration
            << " throughput=" << calls.load() / duration
            << " p95_including_wait=" << all[static_cast<size_t>((all.size() - 1) * 0.95)] << " ms\n";
        assert(CV_Release(handle) == AI_OK);
#if defined(AIENGINE_CV_TEST_HOOKS)
        const CVTestStats stats = CVTest_Stats();
        assert(stats.active == 0 && stats.peak_active <= stats.limit &&
            stats.idle_bytes <= stats.budget && stats.live_templates == 0);
        std::cout << "pool_limit=" << stats.limit << " peak_active=" << stats.peak_active
            << " idle_bytes=" << stats.idle_bytes << " budget=" << stats.budget << '\n';
#endif
        return 0;
    }
    const double fixed_hit = benchmark(handle, "fixed_hit", "xyq.bmp", constant, iterations);
    const double fixed_miss = benchmark(handle, "fixed_miss", "CV_123.bmp", constant, iterations);
    const double changed_hit = benchmark(handle, "changed_hit", "xyq.bmp", changed, iterations);
    const double changed_miss = benchmark(handle, "changed_miss", "CV_123.bmp", changed, iterations);
    assert(CV_Release(handle) == AI_OK);
    return enforce && std::max({fixed_hit, fixed_miss, changed_hit, changed_miss}) > 5.0 ? 3 : 0;
}
