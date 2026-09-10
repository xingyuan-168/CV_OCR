#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <bcrypt.h>
#include <compressapi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime_bundle_format.h"

namespace {

struct SourceEntry {
    std::string name;
    std::filesystem::path path;
    uint32_t flags = 0;
    std::vector<uint8_t> raw;
    std::vector<uint8_t> compressed;
    std::array<uint8_t, 32> sha256{};
};

std::string windows_error(const char* action, DWORD code = GetLastError()) {
    std::ostringstream out;
    out << action << " (Windows error " << code << ')';
    return out.str();
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open input file: " + path.u8string());
    }
    const std::streamoff length = input.tellg();
    if (length < 0 || static_cast<uint64_t>(length) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Invalid input file size: " + path.u8string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty() &&
        !input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot read input file: " + path.u8string());
    }
    return bytes;
}

std::array<uint8_t, 32> sha256(const void* data, size_t size) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0;
    DWORD result_size = 0;
    std::vector<uint8_t> object;
    std::array<uint8_t, 32> digest{};

    NTSTATUS status = BCryptOpenAlgorithmProvider(
        &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (status < 0) {
        throw std::runtime_error("BCryptOpenAlgorithmProvider failed");
    }
    status = BCryptGetProperty(
        algorithm,
        BCRYPT_OBJECT_LENGTH,
        reinterpret_cast<PUCHAR>(&object_size),
        sizeof(object_size),
        &result_size,
        0);
    if (status < 0 || object_size == 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("BCryptGetProperty failed");
    }
    object.resize(object_size);
    status = BCryptCreateHash(
        algorithm, &hash, object.data(), object_size, nullptr, 0, 0);
    if (status >= 0) {
        const uint8_t* cursor = static_cast<const uint8_t*>(data);
        size_t remaining = size;
        while (remaining > 0 && status >= 0) {
            const ULONG chunk = static_cast<ULONG>(
                std::min<size_t>(remaining, std::numeric_limits<ULONG>::max()));
            status = BCryptHashData(
                hash, const_cast<PUCHAR>(cursor), chunk, 0);
            cursor += chunk;
            remaining -= chunk;
        }
    }
    if (status >= 0) {
        status = BCryptFinishHash(
            hash, digest.data(), static_cast<ULONG>(digest.size()), 0);
    }
    if (hash != nullptr) BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (status < 0) {
        throw std::runtime_error("SHA-256 calculation failed");
    }
    return digest;
}

std::string hex_digest(const std::array<uint8_t, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

std::vector<uint8_t> compress_bytes(
    COMPRESSOR_HANDLE compressor,
    const std::vector<uint8_t>& raw) {
    SIZE_T required = 0;
    static const uint8_t kEmpty = 0;
    const void* input = raw.empty()
        ? static_cast<const void*>(&kEmpty)
        : static_cast<const void*>(raw.data());
    if (!Compress(
            compressor, input, raw.size(), nullptr, 0, &required) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        throw std::runtime_error(windows_error("Compression size query failed"));
    }
    std::vector<uint8_t> output(required);
    SIZE_T written = 0;
    if (!Compress(
            compressor,
            input,
            raw.size(),
            output.empty() ? nullptr : output.data(),
            output.size(),
            &written)) {
        throw std::runtime_error(windows_error("Compression failed"));
    }
    output.resize(written);
    return output;
}

SourceEntry parse_source(const std::string& value, uint32_t flags) {
    const size_t separator = value.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
        throw std::runtime_error("Expected logical-name=path, got: " + value);
    }
    SourceEntry entry;
    entry.name = value.substr(0, separator);
    entry.path = std::filesystem::u8path(value.substr(separator + 1));
    entry.flags = flags;
    if (entry.name.size() >=
            sizeof(static_cast<ai_runtime::BundleEntry*>(nullptr)->name) ||
        entry.name.find('/') != std::string::npos ||
        entry.name.find('\\') != std::string::npos ||
        entry.name == "." ||
        entry.name == "..") {
        throw std::runtime_error("Unsafe or too long logical name: " + entry.name);
    }
    return entry;
}

std::string make_manifest(const std::vector<SourceEntry>& entries) {
    std::ostringstream out;
    out << "{\n"
        << "  \"package_type\": \"embedded-core\",\n"
        << "  \"project_version\": \"0.13.0\",\n"
        << "  \"delivery_version\": \"v22\",\n"
        << "  \"worker_protocol\": 22,\n"
        << "  \"ort_version\": \"1.24.4\",\n"
        << "  \"directml_version\": \"1.15.4\",\n"
        << "  \"compression\": \"XPRESS Huffman\",\n"
        << "  \"providers\": [\"DmlExecutionProvider\", \"CPUExecutionProvider\"],\n"
        << "  \"files\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const SourceEntry& entry = entries[i];
        if (i > 0) out << ",\n";
        out << "    {\"name\":\"" << json_escape(entry.name)
            << "\",\"size\":" << entry.raw.size()
            << ",\"sha256\":\"" << hex_digest(entry.sha256)
            << "\",\"runtime\":" << ((entry.flags & ai_runtime::kEntryRuntime) ? "true" : "false")
            << ",\"directml_optional\":" << ((entry.flags & ai_runtime::kEntryDirectML) ? "true" : "false")
            << ",\"notice\":" << ((entry.flags & ai_runtime::kEntryNotice) ? "true" : "false")
            << '}';
    }
    out << "\n  ]\n}\n";
    return out.str();
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc < 4 || argv == nullptr) {
            std::cerr
                << "usage: cq_runtime_packer output.bin "
                   "(--runtime|--directml|--notice) name=path ...\n";
            return 2;
        }

        const std::filesystem::path output_path(argv[1]);
        std::vector<SourceEntry> entries;
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::wstring kind(argv[i]);
            const std::string value = std::filesystem::path(argv[i + 1]).u8string();
            uint32_t flags = 0;
            if (kind == L"--runtime") {
                flags = ai_runtime::kEntryRuntime;
            } else if (kind == L"--directml") {
                flags = ai_runtime::kEntryRuntime | ai_runtime::kEntryDirectML;
            } else if (kind == L"--notice") {
                flags = ai_runtime::kEntryNotice;
            } else {
                throw std::runtime_error("Unknown entry kind");
            }
            entries.push_back(parse_source(value, flags));
        }

        if ((argc - 2) % 2 != 0) {
            throw std::runtime_error("Missing value for final command-line option");
        }
        if (entries.empty()) {
            throw std::runtime_error("Runtime bundle cannot be empty");
        }
        std::sort(entries.begin(), entries.end(), [](const SourceEntry& a, const SourceEntry& b) {
            return a.name < b.name;
        });
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i > 0 && entries[i - 1].name == entries[i].name) {
                throw std::runtime_error("Duplicate logical name: " + entries[i].name);
            }
            entries[i].raw = read_file(entries[i].path);
            entries[i].sha256 = sha256(entries[i].raw.data(), entries[i].raw.size());
        }

        SourceEntry manifest;
        manifest.name = "runtime-manifest.json";
        manifest.flags = ai_runtime::kEntryMetadata;
        const std::string manifest_text = make_manifest(entries);
        manifest.raw.assign(manifest_text.begin(), manifest_text.end());
        manifest.sha256 = sha256(manifest.raw.data(), manifest.raw.size());
        entries.push_back(std::move(manifest));

        COMPRESSOR_HANDLE compressor = nullptr;
        if (!CreateCompressor(COMPRESS_ALGORITHM_XPRESS_HUFF, nullptr, &compressor)) {
            throw std::runtime_error(windows_error("CreateCompressor failed"));
        }
        try {
            for (SourceEntry& entry : entries) {
                entry.compressed = compress_bytes(compressor, entry.raw);
            }
        } catch (...) {
            CloseCompressor(compressor);
            throw;
        }
        CloseCompressor(compressor);

        ai_runtime::BundleHeader header{};
        std::memcpy(header.magic, ai_runtime::kBundleMagic, sizeof(header.magic));
        header.format_version = ai_runtime::kBundleFormatVersion;
        header.file_count = static_cast<uint32_t>(entries.size());
        header.index_size = static_cast<uint64_t>(entries.size()) * sizeof(ai_runtime::BundleEntry);

        std::vector<ai_runtime::BundleEntry> index(entries.size());
        uint64_t data_offset = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            const SourceEntry& source = entries[i];
            ai_runtime::BundleEntry& target = index[i];
            std::memcpy(target.name, source.name.data(), source.name.size());
            target.flags = source.flags;
            target.uncompressed_size = source.raw.size();
            target.compressed_size = source.compressed.size();
            target.data_offset = data_offset;
            std::memcpy(target.sha256, source.sha256.data(), source.sha256.size());
            data_offset += target.compressed_size;
        }
        header.data_size = data_offset;

        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot create output file: " + output_path.u8string());
        }
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(
            reinterpret_cast<const char*>(index.data()),
            static_cast<std::streamsize>(index.size() * sizeof(index.front())));
        for (const SourceEntry& entry : entries) {
            if (!entry.compressed.empty()) {
                output.write(
                    reinterpret_cast<const char*>(entry.compressed.data()),
                    static_cast<std::streamsize>(entry.compressed.size()));
            }
        }
        if (!output) {
            throw std::runtime_error("Failed while writing runtime bundle");
        }

        const uint64_t raw_total = [&entries]() {
            uint64_t value = 0;
            for (const SourceEntry& entry : entries) value += entry.raw.size();
            return value;
        }();
        std::cout << "Created " << output_path.u8string()
                  << " with " << entries.size() << " entries; "
                  << raw_total << " bytes raw, "
                  << (sizeof(header) + header.index_size + header.data_size)
                  << " bytes packed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "runtime_packer: " << ex.what() << '\n';
        return 1;
    }
}
