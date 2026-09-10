#include "memory_zip.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using Bytef = unsigned char;
using uInt = unsigned int;
using uLong = unsigned long;
using voidpf = void*;
using alloc_func = voidpf (*)(voidpf, uInt, uInt);
using free_func = void (*)(voidpf, voidpf);

struct z_stream_s {
    Bytef* next_in;
    uInt avail_in;
    uLong total_in;
    Bytef* next_out;
    uInt avail_out;
    uLong total_out;
    char* msg;
    void* state;
    alloc_func zalloc;
    free_func zfree;
    voidpf opaque;
    int data_type;
    uLong adler;
    uLong reserved;
};

extern "C" {
const char* zlibVersion(void);
int inflateInit2_(z_stream_s* stream, int window_bits, const char* version, int stream_size);
int inflate(z_stream_s* stream, int flush);
int inflateEnd(z_stream_s* stream);
uLong crc32(uLong crc, const Bytef* data, uInt size);
}

constexpr uint32_t kLocalHeaderSignature = 0x04034b50u;
constexpr uint32_t kCentralHeaderSignature = 0x02014b50u;
constexpr uint32_t kEndSignature = 0x06054b50u;
constexpr size_t kEndHeaderSize = 22;
constexpr size_t kCentralHeaderSize = 46;
constexpr size_t kLocalHeaderSize = 30;
constexpr size_t kMaxEndSearch = 65535 + kEndHeaderSize;
constexpr uint16_t kUtf8Flag = 1u << 11;
constexpr uint16_t kEncryptedFlag = 1u << 0;
constexpr uint16_t kStrongEncryptedFlag = 1u << 6;
constexpr uint16_t kStoredMethod = 0;
constexpr uint16_t kDeflateMethod = 8;
constexpr uint32_t kMaxEntries = 4096;
constexpr uint64_t kMaxSingleUncompressed = 64ull * 1024 * 1024;
constexpr uint64_t kMaxTotalUncompressed = 512ull * 1024 * 1024;
constexpr uint64_t kMaxCompressionRatio = 200;
constexpr int kZFinish = 4;
constexpr int kZOk = 0;
constexpr int kZStreamEnd = 1;

uint16_t read_u16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
        static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

uint32_t read_u32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);
}

bool range_inside(size_t offset, size_t length, size_t total) {
    return offset <= total && length <= total - offset;
}

bool fail(std::string* error, const std::string& detail) {
    if (error != nullptr) *error = detail;
    return false;
}

bool safe_archive_path(const std::string& raw_name) {
    if (raw_name.empty() || raw_name.find('\0') != std::string::npos) return false;
    std::string name = raw_name;
    std::replace(name.begin(), name.end(), '\\', '/');
    if (name.empty() || name.front() == '/') return false;
    if (name.size() >= 2 && name[1] == ':') return false;

    const bool directory = name.back() == '/';
    size_t begin = 0;
    while (begin <= name.size()) {
        const size_t end = name.find('/', begin);
        const std::string component = name.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const bool trailing_directory_component = directory && component.empty() &&
            end == std::string::npos && begin == name.size();
        if (!trailing_directory_component &&
            (component.empty() || component == "." || component == "..")) {
            return false;
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return true;
}

bool is_bmp_entry(const std::string& raw_name) {
    const size_t slash = raw_name.find_last_of("/\\");
    const size_t basename = slash == std::string::npos ? 0 : slash + 1;
    if (basename >= raw_name.size() || raw_name.size() - basename < 5) return false;
    const size_t dot = raw_name.size() - 4;
    if (dot < basename || raw_name[dot] != '.') return false;
    const auto lower_ascii = [](char ch) {
        return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch - 'A' + 'a') : ch;
    };
    return lower_ascii(raw_name[dot + 1]) == 'b' &&
        lower_ascii(raw_name[dot + 2]) == 'm' &&
        lower_ascii(raw_name[dot + 3]) == 'p';
}

bool inflate_raw(
    const uint8_t* compressed,
    uint32_t compressed_size,
    uint8_t* output,
    uint32_t output_size,
    std::string* error) {
    z_stream_s stream{};
    stream.next_in = const_cast<Bytef*>(compressed);
    stream.avail_in = compressed_size;
    stream.next_out = output;
    stream.avail_out = output_size;
    const char* version = zlibVersion();
    if (version == nullptr || inflateInit2_(&stream, -15, version, static_cast<int>(sizeof(stream))) != kZOk) {
        return fail(error, "failed to initialize raw DEFLATE decoder");
    }
    const int status = inflate(&stream, kZFinish);
    const bool complete = status == kZStreamEnd && stream.total_in == compressed_size &&
        stream.total_out == output_size;
    inflateEnd(&stream);
    if (!complete) return fail(error, "invalid or truncated DEFLATE stream");
    return true;
}

bool verify_crc(const std::vector<uint8_t>& data, uint32_t expected) {
    uLong actual = crc32(0, nullptr, 0);
    if (!data.empty()) {
        actual = crc32(actual, data.data(), static_cast<uInt>(data.size()));
    }
    return static_cast<uint32_t>(actual) == expected;
}

} // namespace

namespace ai {

bool extract_memory_zip(
    const uint8_t* zip_data,
    int32_t zip_size,
    std::vector<MemoryZipEntry>* entries,
    std::string* error) {
    if (entries == nullptr || zip_data == nullptr || zip_size < static_cast<int32_t>(kEndHeaderSize)) {
        return fail(error, "ZIP data is empty or too small");
    }
    entries->clear();
    if (error != nullptr) error->clear();
    const size_t total_size = static_cast<size_t>(zip_size);

    const size_t search_begin = total_size > kMaxEndSearch ? total_size - kMaxEndSearch : 0;
    size_t end_offset = std::numeric_limits<size_t>::max();
    for (size_t pos = total_size - kEndHeaderSize;; --pos) {
        if (read_u32(zip_data + pos) == kEndSignature) {
            const uint16_t comment_size = read_u16(zip_data + pos + 20);
            if (range_inside(pos, kEndHeaderSize + comment_size, total_size) &&
                pos + kEndHeaderSize + comment_size == total_size) {
                end_offset = pos;
                break;
            }
        }
        if (pos == search_begin) break;
    }
    if (end_offset == std::numeric_limits<size_t>::max()) {
        return fail(error, "ZIP end-of-central-directory record was not found");
    }

    const uint16_t disk_number = read_u16(zip_data + end_offset + 4);
    const uint16_t central_disk = read_u16(zip_data + end_offset + 6);
    const uint16_t disk_entries = read_u16(zip_data + end_offset + 8);
    const uint16_t total_entries = read_u16(zip_data + end_offset + 10);
    const uint32_t central_size = read_u32(zip_data + end_offset + 12);
    const uint32_t central_offset = read_u32(zip_data + end_offset + 16);
    if (disk_number != 0 || central_disk != 0 || disk_entries != total_entries) {
        return fail(error, "multi-disk ZIP archives are not supported");
    }
    if (total_entries > kMaxEntries) return fail(error, "ZIP contains too many entries");
    if (central_size == std::numeric_limits<uint32_t>::max() ||
        central_offset == std::numeric_limits<uint32_t>::max()) {
        return fail(error, "ZIP64 archives are not supported");
    }
    if (!range_inside(central_offset, central_size, total_size) ||
        static_cast<uint64_t>(central_offset) + central_size > end_offset) {
        return fail(error, "ZIP central directory is outside the archive");
    }

    size_t cursor = central_offset;
    uint64_t total_uncompressed = 0;
    entries->reserve(total_entries);
    for (uint32_t index = 0; index < total_entries; ++index) {
        if (!range_inside(cursor, kCentralHeaderSize, total_size) ||
            read_u32(zip_data + cursor) != kCentralHeaderSignature) {
            return fail(error, "ZIP central directory entry is invalid");
        }
        const uint16_t flags = read_u16(zip_data + cursor + 8);
        const uint16_t method = read_u16(zip_data + cursor + 10);
        const uint32_t expected_crc = read_u32(zip_data + cursor + 16);
        const uint32_t compressed_size = read_u32(zip_data + cursor + 20);
        const uint32_t uncompressed_size = read_u32(zip_data + cursor + 24);
        const uint16_t name_size = read_u16(zip_data + cursor + 28);
        const uint16_t extra_size = read_u16(zip_data + cursor + 30);
        const uint16_t comment_size = read_u16(zip_data + cursor + 32);
        const uint16_t start_disk = read_u16(zip_data + cursor + 34);
        const uint32_t local_offset = read_u32(zip_data + cursor + 42);
        const size_t variable_size = static_cast<size_t>(name_size) + extra_size + comment_size;
        if (!range_inside(cursor + kCentralHeaderSize, variable_size, total_size)) {
            return fail(error, "ZIP central directory variable fields are truncated");
        }
        if ((flags & (kEncryptedFlag | kStrongEncryptedFlag)) != 0) {
            return fail(error, "encrypted ZIP entries are not supported");
        }
        if (start_disk != 0 || compressed_size == std::numeric_limits<uint32_t>::max() ||
            uncompressed_size == std::numeric_limits<uint32_t>::max() ||
            local_offset == std::numeric_limits<uint32_t>::max()) {
            return fail(error, "ZIP64 or multi-disk entries are not supported");
        }

        const std::string raw_name(
            reinterpret_cast<const char*>(zip_data + cursor + kCentralHeaderSize), name_size);
        if (!safe_archive_path(raw_name)) return fail(error, "ZIP entry contains an unsafe path");

        const bool directory = !raw_name.empty() &&
            (raw_name.back() == '/' || raw_name.back() == '\\');
        const bool bmp = !directory && is_bmp_entry(raw_name);
        if (bmp) {
            if (method != kStoredMethod && method != kDeflateMethod) {
                return fail(error, "BMP entry uses an unsupported compression method");
            }
            if (uncompressed_size > kMaxSingleUncompressed) {
                return fail(error, "ZIP entry exceeds the per-file decompression limit");
            }
            if (uncompressed_size > 0 && compressed_size == 0) {
                return fail(error, "ZIP entry has an invalid compression ratio");
            }
            if (compressed_size > 0 &&
                static_cast<uint64_t>(uncompressed_size) >
                    static_cast<uint64_t>(compressed_size) * kMaxCompressionRatio) {
                return fail(error, "ZIP entry exceeds the compression-ratio limit");
            }
            total_uncompressed += uncompressed_size;
            if (total_uncompressed > kMaxTotalUncompressed) {
                return fail(error, "ZIP exceeds the total decompression limit");
            }

            if (!range_inside(local_offset, kLocalHeaderSize, total_size) ||
                read_u32(zip_data + local_offset) != kLocalHeaderSignature) {
                return fail(error, "ZIP local file header is invalid");
            }
            const uint16_t local_flags = read_u16(zip_data + local_offset + 6);
            const uint16_t local_method = read_u16(zip_data + local_offset + 8);
            const uint16_t local_name_size = read_u16(zip_data + local_offset + 26);
            const uint16_t local_extra_size = read_u16(zip_data + local_offset + 28);
            if (local_flags != flags || local_method != method || local_name_size != name_size ||
                !range_inside(local_offset + kLocalHeaderSize,
                    static_cast<size_t>(local_name_size) + local_extra_size, total_size) ||
                std::memcmp(zip_data + local_offset + kLocalHeaderSize, raw_name.data(), name_size) != 0) {
                return fail(error, "ZIP local and central headers do not match");
            }
            const size_t data_offset = local_offset + kLocalHeaderSize + local_name_size + local_extra_size;
            if (!range_inside(data_offset, compressed_size, total_size) ||
                static_cast<uint64_t>(data_offset) + compressed_size > central_offset) {
                return fail(error, "ZIP entry data is outside the file-data area");
            }

            MemoryZipEntry extracted;
            extracted.name_bytes = raw_name;
            extracted.utf8_name = (flags & kUtf8Flag) != 0;
            extracted.data.resize(uncompressed_size);
            if (method == kStoredMethod) {
                if (compressed_size != uncompressed_size) {
                    return fail(error, "stored ZIP entry has mismatched sizes");
                }
                if (compressed_size > 0) {
                    std::memcpy(extracted.data.data(), zip_data + data_offset, compressed_size);
                }
            } else if (!inflate_raw(
                    zip_data + data_offset,
                    compressed_size,
                    extracted.data.data(),
                    uncompressed_size,
                    error)) {
                return false;
            }
            if (!verify_crc(extracted.data, expected_crc)) {
                return fail(error, "ZIP entry CRC check failed");
            }
            entries->push_back(std::move(extracted));
        }

        cursor += kCentralHeaderSize + variable_size;
    }
    if (cursor != static_cast<size_t>(central_offset) + central_size) {
        return fail(error, "ZIP central directory size does not match its entries");
    }
    return true;
}

} // namespace ai
