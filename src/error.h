#pragma once

#include <string>

namespace ai {

// 保存当前线程最近错误文本，同时清除后备错误。
void set_last_error(std::string message);

// 返回当前线程普通错误文本；C ABI 使用包含后备存储的 last_error_c_str。
const std::string& last_error();

// DLL 异常边界专用：不分配内存，截断过长详情，永不抛出。
void set_last_error_fallback(const char* operation, const char* detail) noexcept;
const char* last_error_c_str() noexcept;

} // namespace ai
