#pragma once

#include <chrono>
#include <stdint.h>

namespace ai {

// RAII 耗时统计工具。在一段工作开始前构造，析构时将耗时微秒写入 output_us。
class ScopedLatency {
public:
    // 构造时立即开始计时。
    explicit ScopedLatency(int64_t* output_us) : output_us_(output_us), start_(std::chrono::steady_clock::now()) {}

    // 离开作用域时写入耗时。
    ~ScopedLatency() {
        if (output_us_ == nullptr) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        *output_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }

private:
    // 耗时写入目标；计时可选时允许为空。
    int64_t* output_us_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace ai
