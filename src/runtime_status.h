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
};

void set_runtime_status(RuntimeStatus status);
RuntimeStatus get_runtime_status();
RuntimeStatus get_thread_runtime_status();

} // namespace ai
