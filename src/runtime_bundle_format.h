#pragma once

#include <cstdint>

namespace ai_runtime {

static constexpr char kBundleMagic[8] = {'C', 'Q', 'R', 'T', 'V', '2', '2', '\0'};
static constexpr uint32_t kBundleFormatVersion = 1;
static constexpr uint32_t kRuntimeBundleResourceId = 106;
static constexpr uint32_t kEntryRuntime = 1u << 0;
static constexpr uint32_t kEntryDirectML = 1u << 1;
static constexpr uint32_t kEntryNotice = 1u << 2;
static constexpr uint32_t kEntryMetadata = 1u << 3;

#pragma pack(push, 1)
struct BundleHeader {
    char magic[8];
    uint32_t format_version;
    uint32_t file_count;
    uint64_t index_size;
    uint64_t data_size;
};

struct BundleEntry {
    char name[96];
    uint32_t flags;
    uint32_t reserved;
    uint64_t uncompressed_size;
    uint64_t compressed_size;
    uint64_t data_offset;
    uint8_t sha256[32];
};
#pragma pack(pop)

} // namespace ai_runtime
