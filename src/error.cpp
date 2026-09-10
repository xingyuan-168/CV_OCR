#include "error.h"

#include <utility>

namespace ai {

namespace {
// 每个调用线程保存自己的最近错误。这样可避免一个线程推理失败后，
// 在另一个线程调用 AI_GetLastError() 前覆盖其错误文本。
thread_local std::string g_last_error;
}

// 替换当前线程的最近错误文本。
void set_last_error(std::string message) {
    g_last_error = std::move(message);
}

// 返回当前线程的最近错误文本。
const std::string& last_error() {
    return g_last_error;
}

} // namespace ai
