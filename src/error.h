#pragma once

#include <string>

namespace ai {

// 保存 AI_GetLastError() 使用的进程内最近错误文本。
void set_last_error(std::string message);

// 返回进程内最近一次错误文本。
const std::string& last_error();

} // namespace ai
