#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

struct MemoryZipEntry {
    std::string name_bytes;
    bool utf8_name = false;
    std::vector<uint8_t> data;
};

// Extracts BMP regular files from a single-disk, non-ZIP64 archive.  Directory
// and non-BMP entries are structurally inspected but not decompressed.  The
// caller decodes UTF-8/legacy names and validates BMP content.  No archive
// content is ever written to disk.
bool extract_memory_zip(
    const uint8_t* zip_data,
    int32_t zip_size,
    std::vector<MemoryZipEntry>* entries,
    std::string* error);

} // namespace ai
