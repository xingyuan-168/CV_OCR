#include "config.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ai {

namespace {

// 解析键和值之前，去掉首尾空白字符。
std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

// 比较前统一 TRUE/Yes/ON 等布尔文本的大小写。
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

#if defined(_WIN32)
// 将 C ABI 传入的 UTF-8 路径转为 Windows 宽字符路径，确保中文目录和模型/配置文件名可正常打开。
std::wstring utf8_to_wide(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return std::wstring();
    }

    const int required = MultiByteToWideChar(CP_UTF8, 0, value, -1, nullptr, 0);
    if (required <= 0) {
        return std::wstring();
    }

    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value, -1, &output[0], required);
    if (!output.empty() && output.back() == L'\0') {
        output.pop_back();
    }
    return output;
}
#endif

std::filesystem::path executable_directory() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size()) {
        path.resize(length);
        return std::filesystem::path(path).parent_path();
    }
#endif
    std::error_code error;
    return std::filesystem::current_path(error);
}

std::string resolve_config_resource_path(const std::string& value) {
    if (value.empty()) return value;
    try {
        std::filesystem::path path;
#if defined(_WIN32)
        const std::wstring wide_path = utf8_to_wide(value.c_str());
        if (wide_path.empty()) return value;
        path = std::filesystem::path(wide_path);
#else
        path = std::filesystem::u8path(value);
#endif
        if (!path.is_absolute()) path = executable_directory() / path;
        return path.lexically_normal().u8string();
    } catch (...) {
        return value;
    }
}

} // namespace

// 加载 INI 风格 key=value 文件；空路径表示“使用默认值”。
bool Config::load_file(const char* path, std::string* error) {
    // 加载新文件会完整替换之前的配置状态。
    values_.clear();

    if (path == nullptr || path[0] == '\0') {
        return true;
    }

    std::ifstream input;
    std::filesystem::path resolved_path;
#if defined(_WIN32)
    const std::wstring wide_path = utf8_to_wide(path);
    if (!wide_path.empty()) {
        resolved_path = std::filesystem::path(wide_path);
    } else {
        resolved_path = std::filesystem::path(path);
    }
#else
    resolved_path = std::filesystem::path(path);
#endif
    if (!resolved_path.is_absolute()) resolved_path = executable_directory() / resolved_path;
    input.open(resolved_path.lexically_normal());
    if (!input) {
        if (error != nullptr) {
            *error = std::string("Cannot open config file: ") + path;
        }
        return false;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            if (error != nullptr) {
                std::ostringstream oss;
                oss << "Invalid config line " << line_number << ": " << line;
                *error = oss.str();
            }
            return false;
        }

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        if (key.empty()) {
            if (error != nullptr) {
                std::ostringstream oss;
                oss << "Empty config key at line " << line_number;
                *error = oss.str();
            }
            return false;
        }
        values_[key] = value;
    }

    for (auto& item : values_) {
        const std::string& key = item.first;
        if (key.size() >= 5 && key.compare(key.size() - 5, 5, "_path") == 0) {
            item.second = resolve_config_resource_path(item.second);
        }
    }

    return true;
}

// 按传入内容原样保存字符串值。
void Config::set_string(const std::string& key, const std::string& value) {
    values_[key] = value;
}

// 将整数值保存到与文件加载值相同的字符串映射中。
void Config::set_int(const std::string& key, int32_t value) {
    values_[key] = std::to_string(value);
}

// 返回配置中的字符串值；键不存在时返回 fallback。
std::string Config::get_string(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
        return fallback;
    }
    return it->second;
}

// 解析 int32_t 值；配置内容非法时返回 fallback，不抛异常。
int32_t Config::get_int(const std::string& key, int32_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
        return fallback;
    }
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

// 解析 float 值；配置内容非法时返回 fallback，不抛异常。
float Config::get_float(const std::string& key, float fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
        return fallback;
    }
    try {
        return std::stof(it->second);
    } catch (...) {
        return fallback;
    }
}

// 解析 INI 文件中常见的人类可读布尔写法。
bool Config::get_bool(const std::string& key, bool fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
        return fallback;
    }

    const std::string value = to_lower(trim(it->second));
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return fallback;
}

} // namespace ai
