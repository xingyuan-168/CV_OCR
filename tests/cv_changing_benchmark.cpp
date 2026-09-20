#include "ai_engine.h"
#include "compact_result_test_utils.h"
#if defined(AIENGINE_CV_TEST_HOOKS)
#include "cv_test_hooks.h"
#endif
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
using Bytes = std::vector<uint8_t>;
Bytes read(const std::filesystem::path &p) {
  std::ifstream stream(p, std::ios::binary);
  return Bytes(std::istreambuf_iterator<char>(stream), {});
}
uint32_t word(const Bytes &b, int offset) {
  uint32_t n;
  std::memcpy(&n, b.data() + offset, 4);
  return n;
}
struct Bmp {
  Bytes bytes;
  int w, h, stride, offset;
  bool bottom;
  explicit Bmp(Bytes b) : bytes(std::move(b)) {
    assert(bytes.size() > 54 && bytes[28] == 24);
    w = static_cast<int>(word(bytes, 18));
    const int height = static_cast<int>(word(bytes, 22));
    bottom = height > 0;
    h = std::abs(height);
    stride = (w * 3 + 3) & ~3;
    offset = static_cast<int>(word(bytes, 10));
  }
  uint8_t *row(int y) {
    return bytes.data() + offset + (bottom ? h - 1 - y : y) * stride;
  }
};
int main(int argc, char **argv) {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
  const auto root =
      std::filesystem::absolute(argc > 1 ? argv[1] : "tests/fixtures/cv/base");
  const int iterations = argc > 2 ? std::max(1000, std::atoi(argv[2])) : 1000;
  const bool reference_timing =
      argc > 3 && std::string(argv[3]) == "--reference";
  const int thread_count =
      argc > 4 && std::string(argv[3]) == "--threads" ? std::atoi(argv[4]) : 0;
  assert(thread_count >= 0 && thread_count <= 64);
  Bmp original(read(root / std::filesystem::u8path(u8"大图1.bmp")));
  Bmp first(read(root / std::filesystem::u8path(u8"电.bmp")));
  Bmp second(read(root / std::filesystem::u8path(u8"游.bmp")));
  assert(original.w == 610 && original.h == 318);
  int handle = 0;
  assert(CV_Create(&handle) == AI_OK);
  assert(CV_LoadTemplateDir(handle, root.u8string().c_str(), 0) >= 2);
  struct Frame {
    Bytes bmp;
    float threshold;
    std::string expected[2];
  };
  std::vector<Frame> frames;
  for (int f = 0; f < 64; ++f) {
    Bmp frame(original.bytes);
    for (int y = 0; y < frame.h; ++y) {
      auto *dst = frame.row(y);
      const auto *src = original.row((y + f * 3) % original.h);
      for (int x = 0; x < frame.w; ++x) {
        const int from = (x + f * 7) % frame.w;
        for (int c = 0; c < 3; ++c)
          dst[x * 3 + c] = src[from * 3 + c];
      }
    }
    // Distinct changing backgrounds, explicit misses, and a perturbed target
    // with a threshold on either side of its exact measured score.
    if (f % 8 == 0)
      for (int y = 0; y < frame.h; ++y)
        std::fill(frame.row(y), frame.row(y) + frame.w * 3,
                  static_cast<uint8_t>(f));
    float threshold = 0.4f;
    if (f % 8 == 1 || f % 8 == 2) {
      for (int y = 0; y < frame.h; ++y)
        std::fill(frame.row(y), frame.row(y) + frame.w * 3,
                  static_cast<uint8_t>(f));
      for (int y = 0; y < first.h; ++y)
        std::memcpy(frame.row(y + 31) + 83 * 3, first.row(y), first.w * 3);
      frame.row(33)[85 * 3] ^= 127;
      CVMatchResult match{};
      assert(CV_FindOne(handle, u8"电.bmp", frame.bytes.data(),
                        static_cast<int>(frame.bytes.size()), 0.1f, 0,
                        &match) == 1);
      threshold = f % 16 < 8
                      ? std::nextafter(match.sim, f % 8 == 1 ? 0.0f : 1.0f)
                      : std::clamp(match.sim + (f % 8 == 1 ? -2e-5f : 2e-5f),
                                   0.0f, 1.0f);
    }
    frames.push_back({std::move(frame.bytes), threshold, {}});
  }
  const auto find = [&](Frame &f, bool transparent) {
    const char *result =
        transparent
            ? CV_FindTransparentMultiText(
                  handle, u8"电.bmp|游.bmp", f.bmp.data(),
                  static_cast<int>(f.bmp.size()), "", f.threshold, "FF00FF")
            : CV_FindMultiText(handle, u8"电.bmp|游.bmp", f.bmp.data(),
                               static_cast<int>(f.bmp.size()), "", f.threshold,
                               0);
    assert(result && AI_GetLastError()[0] == '\0');
    return std::string(result);
  };
  for (auto &f : frames)
    for (int transparent = 0; transparent < 2; ++transparent) {
#if defined(AIENGINE_CV_TEST_HOOKS)
      CVTest_Reference(1);
#endif
      f.expected[transparent] = find(f, transparent != 0);
#if defined(AIENGINE_CV_TEST_HOOKS)
      CVTest_Reference(0);
#endif
      const auto actual = find(f, transparent != 0);
      if (actual != f.expected[transparent]) {
        std::cerr << "reference mismatch at frame " << (&f - frames.data())
                  << " threshold=" << f.threshold
                  << "\nexpected=" << f.expected[transparent]
                  << "\nactual=" << actual << '\n';
        return 2;
      }
    }
  if (thread_count) {
    // The full request includes workspace queueing. This is a throughput and
    // stability measurement, separate from the single-caller 5 ms gate below.
    std::atomic<bool> go{false};
    std::atomic<int> ready{0};
    std::vector<std::vector<double>> timings(thread_count);
    std::vector<std::thread> threads;
    for (int t = 0; t < thread_count; ++t) {
      timings[t].reserve(iterations);
      threads.emplace_back([&, t] {
        for (int i = 0; i < 64; ++i) {
          auto &f = frames[(i + t * 7) % frames.size()];
          assert(find(f, i % 2 != 0) == f.expected[i % 2]);
        }
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire))
          std::this_thread::yield();
        for (int i = 0; i < iterations; ++i) {
          auto &f = frames[(i + t * 7) % frames.size()];
          const auto start = std::chrono::steady_clock::now();
          const auto actual = find(f, i % 2 != 0);
          timings[t].push_back(std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count());
          assert(actual == f.expected[i % 2]);
        }
      });
    }
    while (ready.load(std::memory_order_acquire) != thread_count)
      std::this_thread::yield();
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto &thread : threads)
      thread.join();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::vector<double> samples;
    for (const auto &values : timings)
      samples.insert(samples.end(), values.begin(), values.end());
    std::sort(samples.begin(), samples.end());
    std::cout << "changing_concurrent threads=" << thread_count
              << " calls=" << samples.size() << " seconds=" << seconds
              << " throughput=" << samples.size() / seconds
              << " p95_including_wait_ms="
              << samples[size_t((samples.size() - 1) * .95)]
              << " max_ms=" << samples.back() << '\n';
    assert(CV_Release(handle) == AI_OK);
    return 0;
  }
  bool failed = false;
  for (int transparent = 0; transparent < 2; ++transparent) {
#if defined(AIENGINE_CV_TEST_HOOKS)
    CVTest_Reference(reference_timing ? 1 : 0);
#else
    assert(!reference_timing);
#endif
    for (int i = 0; i < 64; ++i)
      assert(find(frames[i], transparent != 0) ==
             frames[i].expected[transparent]);
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
      auto &frame = frames[i % frames.size()];
      const auto start = std::chrono::steady_clock::now();
      const auto actual = find(frame, transparent != 0);
      samples.push_back(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count());
      assert(actual == frame.expected[transparent]);
    }
    std::sort(samples.begin(), samples.end());
    const auto p95 = samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
    std::cout << (reference_timing ? "reference_" : "")
              << (transparent ? "changing_transparent" : "changing_multi")
              << " samples=" << samples.size()
              << " p50=" << samples[samples.size() / 2] << " p95=" << p95
              << " max=" << samples.back() << " ms\n";
    failed |= p95 > 5.0;
  }
  assert(CV_Release(handle) == AI_OK);
  return failed ? 3 : 0;
}
