#include "runtime_status.h"

#include <mutex>
#include <utility>

namespace ai {

namespace {
std::mutex g_runtime_status_mutex;
RuntimeStatus g_runtime_status{"auto", "cpu", "runtime has not loaded a model", true, "core", "", "", {}};
thread_local RuntimeStatus g_thread_runtime_status{"auto", "cpu", "runtime has not loaded a model", true, "core", "", "", {}};
} // namespace

void set_runtime_status(RuntimeStatus status) {
    g_thread_runtime_status = status;
    std::lock_guard<std::mutex> lock(g_runtime_status_mutex);
    g_runtime_status = std::move(status);
}

RuntimeStatus get_runtime_status() {
    std::lock_guard<std::mutex> lock(g_runtime_status_mutex);
    return g_runtime_status;
}

RuntimeStatus get_thread_runtime_status() {
    return g_thread_runtime_status;
}

} // namespace ai
