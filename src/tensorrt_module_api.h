#pragma once
#include <cstdint>
#include <cstddef>
// Private C ABI. No STL objects or allocator ownership cross module boundaries.
namespace ai_trt {
constexpr uint32_t kAbi = 2;
struct Options {
    uint32_t abi = kAbi;
    int32_t device = 0, fp16 = 0, graph = 0;
    uint64_t workspace = 256ull * 1024 * 1024;
    const wchar_t* cache_directory = nullptr;
};
struct Info {
    uint32_t abi = kAbi;
    int32_t width = 0, height = 0, attrs = 0, candidates = 0;
    int32_t cache_hit = 0, fp16 = 0, graph = 0, driver_version = 0, cuda_version = 12080;
    int64_t build_us = 0;
    char gpu[256]{}, version[32]{}, cache_key[65]{};
    char driver_file_version[64]{}, driver_binary_sha256[65]{};
};
struct Timing { int64_t h2d_us = 0, gpu_us = 0, d2h_us = 0; };
struct Api {
    uint32_t abi;
    void* (*create)(const void*, size_t, const Options*, Info*, char*, size_t);
    void (*destroy)(void*);
    float* (*input)(void*);
    const float* (*output)(void*);
    int32_t (*run)(void*, Timing*, char*, size_t);
};
using GetApi = const Api* (*)(uint32_t);
} // namespace ai_trt
