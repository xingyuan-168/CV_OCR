#pragma once
#include <windows.h>
#include <cstdint>
namespace ai_worker {
inline bool read_frame(HANDLE pipe, void* buffer, DWORD length) {
    auto* bytes = static_cast<uint8_t*>(buffer);
    DWORD total = 0;
    while (total < length) {
        DWORD count = 0;
        if (!ReadFile(pipe, bytes + total, length - total, &count, nullptr) || !count) return false;
        total += count;
    }
    return true;
}
inline bool write_frame(HANDLE pipe, const void* buffer, DWORD length) {
    auto* bytes = static_cast<const uint8_t*>(buffer);
    DWORD total = 0;
    while (total < length) {
        DWORD count = 0;
        if (!WriteFile(pipe, bytes + total, length - total, &count, nullptr) || !count) return false;
        total += count;
    }
    return true;
}
} // namespace ai_worker
