#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <windows.h>
#include <bcrypt.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace ai {
inline std::string sha256(const void* data, size_t length) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    struct Cleanup { BCRYPT_ALG_HANDLE& a; BCRYPT_HASH_HANDLE& h;
        ~Cleanup() { if (h) BCryptDestroyHash(h); if (a) BCryptCloseAlgorithmProvider(a, 0); }
    } cleanup{algorithm, hash};
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) throw std::runtime_error("SHA-256 initialization failed");
    auto* bytes = static_cast<const uint8_t*>(data);
    while (length) {
        const ULONG count = static_cast<ULONG>(std::min<size_t>(length, 0x40000000));
        if (BCryptHashData(hash, const_cast<PUCHAR>(bytes), count, 0) < 0) throw std::runtime_error("SHA-256 update failed");
        bytes += count; length -= count;
    }
    uint8_t digest[32];
    if (BCryptFinishHash(hash, digest, sizeof(digest), 0) < 0) throw std::runtime_error("SHA-256 finish failed");
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (uint8_t b : digest) { result += hex[b >> 4]; result += hex[b & 15]; }
    return result;
}
} // namespace ai
