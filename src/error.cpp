#include "error.h"

#include <utility>

namespace ai {

namespace {
// 每个调用线程保存自己的最近错误。这样可避免一个线程推理失败后，
// 在另一个线程调用 AI_GetLastError() 前覆盖其错误文本。
thread_local std::string g_last_error;
thread_local char g_fallback_error[1024]{};
}

// 替换当前线程的最近错误文本。
void set_last_error(std::string message) {
    g_last_error = std::move(message);
    g_fallback_error[0] = '\0';
}

// 返回当前线程的最近错误文本。
const std::string& last_error() {
    return g_last_error;
}

void set_last_error_fallback(const char* operation, const char* detail) noexcept {
    g_last_error.clear();
    size_t pos = 0;
    for (const char* part : {operation, ": ", detail}) {
        if (part == nullptr) continue;
        while (*part && pos + 1 < sizeof(g_fallback_error)) g_fallback_error[pos++] = *part++;
    }
    g_fallback_error[pos] = '\0';
}

const char* last_error_c_str() noexcept {
    return g_fallback_error[0] ? g_fallback_error : g_last_error.c_str();
}

} // namespace ai
