#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

struct RuntimeStatus {
    std::string requested;
    std::string active;
    std::string reason;
    bool degraded = false;
    std::string runtime_flavor;
    std::string ort_version;
    std::string ort_path;
    std::vector<std::string> available_providers;
    bool mixed_cpu_fallback = false;
    int32_t device_id = 0;
    std::string adapter_name;
    std::string selection_basis;
    std::string calibration_key;
    double cpu_calibration_ms = -1.0;
    double directml_calibration_ms = -1.0;
    std::string precision = "fp32";
    std::string tensorrt_version;
    int32_t cuda_version = 0;
    int32_t driver_version = 0;
    std::string driver_file_version;
    std::string driver_binary_sha256;
    int32_t execution_slots = 0;
    bool engine_cache_hit = false;
    bool cuda_graph = false;
    int64_t engine_build_us = 0;
    std::string engine_cache_key;
};

void set_runtime_status(RuntimeStatus status);
RuntimeStatus get_runtime_status();
RuntimeStatus get_thread_runtime_status();

} // namespace ai
