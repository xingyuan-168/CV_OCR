#pragma once

#include <stdint.h>

#include <string>
#include <unordered_map>

namespace ai {

// 运行时使用的轻量 INI 风格键值存储。
//
// 解析器接受 "key=value" 行，并忽略空行/注释行。所有值都按字符串保存，
// 需要时再由类型化 getter 转换。
class Config {
public:
    // 从路径加载键值对。空指针或空路径视为“空配置”，并返回成功。
    bool load_file(const char* path, std::string* error);

    // 将类型化值写入存储，用于通过公开 API 参数覆盖 runtime.device 等配置。
    void set_string(const std::string& key, const std::string& value);
    void set_int(const std::string& key, int32_t value);

    // 读取值；键不存在或转换失败时返回 fallback。
    std::string get_string(const std::string& key, const std::string& fallback) const;
    int32_t get_int(const std::string& key, int32_t fallback) const;
    float get_float(const std::string& key, float fallback) const;
    bool get_bool(const std::string& key, bool fallback) const;

private:
    std::unordered_map<std::string, std::string> values_;
};

} // namespace ai
