#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ai {
// Per request, not per model: concurrent callers must never query one shared
// "last request" value to attribute their own pipeline latency.
struct YoloTiming {
    uint64_t request_id = 0;
    int64_t pack_us = 0, connect_us = 0, transport_us = 0;
    int64_t bmp_us = 0, wait_us = 0, preprocess_us = 0, run_us = 0;
    int64_t postprocess_us = 0, json_us = 0;
    int64_t h2d_us = 0, gpu_us = 0, d2h_us = 0;
    int64_t worker_us = 0;
};
inline thread_local YoloTiming yolo_timing;
using YoloClock = std::chrono::steady_clock;
inline int64_t elapsed_us(YoloClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(YoloClock::now() - start).count();
}
inline uint64_t next_yolo_request_id() {
    static std::atomic<uint32_t> sequence{1};
#if defined(_WIN32)
    return (static_cast<uint64_t>(GetCurrentProcessId()) << 32) |
        sequence.fetch_add(1, std::memory_order_relaxed);
#else
    return sequence.fetch_add(1, std::memory_order_relaxed);
#endif
}
inline void write_yolo_trace(int64_t total_us, int32_t status) {
    // Separate files per process; records from DLL and Worker join on request_id.
    static const std::string path = [] {
#if defined(_WIN32)
        const DWORD needed=GetEnvironmentVariableW(L"CQ_AI_YOLO_TRACE_PREFIX",nullptr,0);
        if(!needed)return std::string{};
        std::wstring value(needed,0);
        const DWORD size=GetEnvironmentVariableW(L"CQ_AI_YOLO_TRACE_PREFIX",value.data(),needed);
        if(!size ||size>=needed)return std::string{};
        const int bytes=WideCharToMultiByte(CP_UTF8,0,value.data(),size,nullptr,0,nullptr,nullptr);
        std::string prefix(bytes,0);WideCharToMultiByte(CP_UTF8,0,value.data(),size,prefix.data(),bytes,nullptr,nullptr);
        return prefix + "-" + std::to_string(GetCurrentProcessId()) + ".csv";
#else
        const char* prefix = std::getenv("CQ_AI_YOLO_TRACE_PREFIX");
        if (!prefix || !*prefix) return std::string{};
        return std::string(prefix) + ".csv";
#endif
    }();
    if (path.empty()) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::ofstream out(std::filesystem::u8path(path), std::ios::out);
    static bool header = false;
    if (!out) return;
    if (!header) {
        out << "request_id,status,total_us,pack_us,connect_us,transport_us,bmp_us,wait_us,preprocess_us,run_us,postprocess_us,json_us,h2d_us,gpu_us,d2h_us,worker_us\n";
        header = true;
    }
    const auto& t = yolo_timing;
    out << t.request_id << ',' << status << ',' << total_us << ',' << t.pack_us << ',' << t.connect_us << ',' << t.transport_us
        << ',' << t.bmp_us << ',' << t.wait_us << ',' << t.preprocess_us << ',' << t.run_us << ',' << t.postprocess_us
        << ',' << t.json_us << ',' << t.h2d_us << ',' << t.gpu_us << ',' << t.d2h_us << ',' << t.worker_us << '\n';
}
} // namespace ai
