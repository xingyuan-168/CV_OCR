#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "runtime_bundle_loader.h"

#include <windows.h>
#include <bcrypt.h>
#include <compressapi.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "runtime_bundle_format.h"

namespace ai_runtime {
namespace {

struct BundleView {
    const uint8_t* bytes = nullptr;
    size_t size = 0;
    const BundleHeader* header = nullptr;
    const BundleEntry* entries = nullptr;
    const uint8_t* data = nullptr;
};

std::mutex g_state_mutex;
bool g_initialized = false;
bool g_directml_cached = false;
HMODULE g_ort_module = nullptr;
DLL_DIRECTORY_COOKIE g_dll_directory_cookie = nullptr;
std::filesystem::path g_cache_path;
std::string g_bundle_hash;
std::string g_loaded_ort_path;

std::string wide_to_utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(
        CP_UTF8,
        0,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (required <= 0) return {};
    std::string output(static_cast<size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        value.data(),
        static_cast<int>(value.size()),
        output.data(),
        required,
        nullptr,
        nullptr);
    return output;
}

std::string windows_error_text(DWORD code) {
    wchar_t* message = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<wchar_t*>(&message),
        0,
        nullptr);
    std::wstring text;
    if (length > 0 && message != nullptr) {
        text.assign(message, message + length);
        while (!text.empty() &&
               (text.back() == L'\r' || text.back() == L'\n' || text.back() == L' ')) {
            text.pop_back();
        }
    }
    if (message != nullptr) LocalFree(message);
    return wide_to_utf8(text);
}

std::string windows_error(const char* action, DWORD code = GetLastError()) {
    std::ostringstream out;
    out << action << ": Windows error " << code;
    const std::string detail = windows_error_text(code);
    if (!detail.empty()) out << " (" << detail << ')';
    return out.str();
}

bool get_resource_bundle(BundleView* view, std::string* error) {
    if (view == nullptr) return false;
    *view = {};
    HMODULE module = GetModuleHandleW(nullptr);
    HRSRC resource = FindResourceW(
        module,
        MAKEINTRESOURCEW(kRuntimeBundleResourceId),
        MAKEINTRESOURCEW(10));
    if (resource == nullptr) {
        if (error != nullptr) *error = "Embedded runtime package resource is missing";
        return false;
    }
    HGLOBAL loaded = LoadResource(module, resource);
    const DWORD resource_size = SizeofResource(module, resource);
    const void* resource_data = loaded == nullptr ? nullptr : LockResource(loaded);
    if (resource_data == nullptr || resource_size < sizeof(BundleHeader)) {
        if (error != nullptr) *error = "Embedded runtime package is corrupt";
        return false;
    }

    const auto* bytes = static_cast<const uint8_t*>(resource_data);
    const auto* header = reinterpret_cast<const BundleHeader*>(bytes);
    if (std::memcmp(header->magic, kBundleMagic, sizeof(header->magic)) != 0 ||
        header->format_version != kBundleFormatVersion ||
        header->file_count == 0 ||
        header->file_count > 256) {
        if (error != nullptr) *error = "Embedded runtime package has an unsupported format";
        return false;
    }
    const uint64_t expected_index =
        static_cast<uint64_t>(header->file_count) * sizeof(BundleEntry);
    const uint64_t total =
        static_cast<uint64_t>(sizeof(BundleHeader)) +
        header->index_size +
        header->data_size;
    if (header->index_size != expected_index ||
        total != resource_size ||
        total > std::numeric_limits<size_t>::max()) {
        if (error != nullptr) *error = "Embedded runtime package index is corrupt";
        return false;
    }

    const auto* entries = reinterpret_cast<const BundleEntry*>(
        bytes + sizeof(BundleHeader));
    const uint8_t* data = bytes + sizeof(BundleHeader) + header->index_size;
    for (uint32_t i = 0; i < header->file_count; ++i) {
        const BundleEntry& entry = entries[i];
        if (entry.name[0] == '\0' ||
            std::memchr(entry.name, '\0', sizeof(entry.name)) == nullptr ||
            entry.uncompressed_size > std::numeric_limits<size_t>::max() ||
            entry.compressed_size > std::numeric_limits<size_t>::max() ||
            entry.data_offset > header->data_size ||
            entry.compressed_size > header->data_size - entry.data_offset) {
            if (error != nullptr) *error = "Embedded runtime package entry is corrupt";
            return false;
        }
        const std::string name(entry.name);
        if (name == "." || name == ".." ||
            name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos ||
            name.find(':') != std::string::npos) {
            if (error != nullptr) *error = "Embedded runtime package contains an unsafe file name";
            return false;
        }
    }

    view->bytes = bytes;
    view->size = static_cast<size_t>(resource_size);
    view->header = header;
    view->entries = entries;
    view->data = data;
    return true;
}

bool sha256_bytes(const void* data, size_t size, std::array<uint8_t, 32>* digest) {
    if (digest == nullptr) return false;
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0;
    DWORD result_size = 0;
    std::vector<uint8_t> object;

    NTSTATUS status = BCryptOpenAlgorithmProvider(
        &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (status >= 0) {
        status = BCryptGetProperty(
            algorithm,
            BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_size),
            sizeof(object_size),
            &result_size,
            0);
    }
    if (status >= 0 && object_size > 0) {
        object.resize(object_size);
        status = BCryptCreateHash(
            algorithm, &hash, object.data(), object_size, nullptr, 0, 0);
    }
    const uint8_t* cursor = static_cast<const uint8_t*>(data);
    size_t remaining = size;
    while (status >= 0 && remaining > 0) {
        const ULONG chunk = static_cast<ULONG>(
            std::min<size_t>(remaining, std::numeric_limits<ULONG>::max()));
        status = BCryptHashData(
            hash, const_cast<PUCHAR>(cursor), chunk, 0);
        cursor += chunk;
        remaining -= chunk;
    }
    if (status >= 0) {
        status = BCryptFinishHash(
            hash, digest->data(), static_cast<ULONG>(digest->size()), 0);
    }
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    return status >= 0;
}

std::string hex_digest(const uint8_t* digest, size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        out << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return out.str();
}

bool digest_matches(const std::vector<uint8_t>& bytes, const BundleEntry& entry) {
    std::array<uint8_t, 32> digest{};
    return sha256_bytes(bytes.data(), bytes.size(), &digest) &&
        std::memcmp(digest.data(), entry.sha256, digest.size()) == 0;
}

bool decompress_entry(
    const BundleView& bundle,
    const BundleEntry& entry,
    std::vector<uint8_t>* output,
    std::string* error) {
    if (output == nullptr) return false;
    output->assign(static_cast<size_t>(entry.uncompressed_size), 0);
    DECOMPRESSOR_HANDLE decompressor = nullptr;
    if (!CreateDecompressor(
            COMPRESS_ALGORITHM_XPRESS_HUFF, nullptr, &decompressor)) {
        if (error != nullptr) *error = windows_error("CreateDecompressor failed");
        return false;
    }
    SIZE_T written = 0;
    const BOOL ok = Decompress(
        decompressor,
        bundle.data + entry.data_offset,
        static_cast<SIZE_T>(entry.compressed_size),
        output->empty() ? nullptr : output->data(),
        output->size(),
        &written);
    const DWORD failure = ok ? ERROR_SUCCESS : GetLastError();
    CloseDecompressor(decompressor);
    if (!ok || written != output->size()) {
        if (error != nullptr) {
            *error = ok
                ? "Embedded runtime entry decompressed to an unexpected size"
                : windows_error("Embedded runtime decompression failed", failure);
        }
        output->clear();
        return false;
    }
    if (!digest_matches(*output, entry)) {
        if (error != nullptr) {
            *error = std::string("SHA-256 mismatch for embedded ") + entry.name;
        }
        output->clear();
        return false;
    }
    return true;
}

const BundleEntry* find_entry(
    const BundleView& bundle,
    const char* name) {
    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        if (std::strcmp(bundle.entries[i].name, name) == 0) {
            return &bundle.entries[i];
        }
    }
    return nullptr;
}

bool read_file(const std::filesystem::path& path, std::vector<uint8_t>* bytes) {
    if (bytes == nullptr) return false;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff length = input.tellg();
    if (length < 0 ||
        static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max()) {
        return false;
    }
    bytes->assign(static_cast<size_t>(length), 0);
    input.seekg(0, std::ios::beg);
    return bytes->empty() ||
        static_cast<bool>(input.read(
            reinterpret_cast<char*>(bytes->data()),
            static_cast<std::streamsize>(bytes->size())));
}

bool cached_file_valid(
    const std::filesystem::path& directory,
    const BundleEntry& entry) {
    std::vector<uint8_t> bytes;
    return read_file(directory / std::filesystem::u8path(entry.name), &bytes) &&
        bytes.size() == entry.uncompressed_size &&
        digest_matches(bytes, entry);
}

std::filesystem::path local_appdata_path(std::string* error) {
    DWORD required = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (required == 0) {
        if (error != nullptr) *error = "LOCALAPPDATA is unavailable";
        return {};
    }
    std::wstring value(required, L'\0');
    const DWORD written = GetEnvironmentVariableW(
        L"LOCALAPPDATA", value.data(), required);
    if (written == 0 || written >= required) {
        if (error != nullptr) *error = windows_error("Cannot read LOCALAPPDATA");
        return {};
    }
    value.resize(written);
    return std::filesystem::path(value);
}

class ScopedMutex {
public:
    ~ScopedMutex() {
        if (acquired_ && handle_ != nullptr) ReleaseMutex(handle_);
        if (handle_ != nullptr) CloseHandle(handle_);
    }

    bool acquire(const std::wstring& name, std::string* error) {
        handle_ = CreateMutexW(nullptr, FALSE, name.c_str());
        if (handle_ == nullptr) {
            if (error != nullptr) *error = windows_error("Cannot create runtime cache mutex");
            return false;
        }
        const DWORD wait = WaitForSingleObject(handle_, 120000);
        if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) {
            if (error != nullptr) *error = "Timed out waiting for runtime cache mutex";
            return false;
        }
        acquired_ = true;
        return true;
    }

private:
    HANDLE handle_ = nullptr;
    bool acquired_ = false;
};

bool write_file(
    const std::filesystem::path& path,
    const std::vector<uint8_t>& bytes,
    std::string* error) {
    HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        if (error != nullptr) {
            *error = "Cannot create runtime cache file " +
                path.filename().u8string() + ": " +
                windows_error_text(GetLastError());
        }
        return false;
    }
    size_t offset = 0;
    bool ok = true;
    while (offset < bytes.size()) {
        const DWORD chunk = static_cast<DWORD>(
            std::min<size_t>(
                bytes.size() - offset,
                std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!WriteFile(
                file, bytes.data() + offset, chunk, &written, nullptr) ||
            written != chunk) {
            ok = false;
            if (error != nullptr) {
                *error = "Failed to write runtime cache file " +
                    path.filename().u8string() + ": " +
                    windows_error_text(GetLastError());
            }
            break;
        }
        offset += written;
    }
    if (ok && !FlushFileBuffers(file)) {
        ok = false;
        if (error != nullptr) {
            *error = "Failed to flush runtime cache file " +
                path.filename().u8string();
        }
    }
    CloseHandle(file);
    return ok;
}

bool enough_disk_space(
    const std::filesystem::path& path,
    uint64_t required,
    std::string* error) {
    ULARGE_INTEGER available{};
    if (!GetDiskFreeSpaceExW(path.c_str(), &available, nullptr, nullptr)) {
        if (error != nullptr) *error = windows_error("Cannot query runtime cache free space");
        return false;
    }
    constexpr uint64_t kReserve = 1024u * 1024u;
    if (available.QuadPart < required + kReserve) {
        if (error != nullptr) *error = "Runtime cache has insufficient free space";
        return false;
    }
    return true;
}

bool is_baseline_runtime_entry(const BundleEntry& entry) {
    return (entry.flags & kEntryRuntime) != 0 &&
        (entry.flags & kEntryDirectML) == 0;
}

bool cache_baseline_valid(
    const BundleView& bundle,
    const std::filesystem::path& cache) {
    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        const BundleEntry& entry = bundle.entries[i];
        if (is_baseline_runtime_entry(entry) &&
            !cached_file_valid(cache, entry)) {
            return false;
        }
    }
    return true;
}

bool remove_generated_tree(
    const std::filesystem::path& path,
    const std::filesystem::path& expected_parent) {
    if (path.empty() || path.parent_path() != expected_parent) return false;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    return !ec;
}

bool build_baseline_cache(
    const BundleView& bundle,
    const std::filesystem::path& cache,
    std::string* error) {
    if (cache_baseline_valid(bundle, cache)) return true;

    const std::filesystem::path parent = cache.parent_path();
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
        if (error != nullptr) *error = "Cannot create runtime cache directory: " + ec.message();
        return false;
    }

    uint64_t required = 0;
    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        if (is_baseline_runtime_entry(bundle.entries[i])) {
            required += bundle.entries[i].uncompressed_size;
        }
    }
    if (!enough_disk_space(parent, required, error)) return false;

    const std::wstring suffix =
        L".tmp." + std::to_wstring(GetCurrentProcessId()) +
        L"." + std::to_wstring(GetTickCount64());
    const std::filesystem::path temporary =
        parent / std::filesystem::path(cache.filename().wstring() + suffix);
    remove_generated_tree(temporary, parent);
    std::filesystem::create_directories(temporary, ec);
    if (ec) {
        if (error != nullptr) *error = "Cannot create runtime cache temporary directory: " + ec.message();
        return false;
    }

    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        const BundleEntry& entry = bundle.entries[i];
        if (!is_baseline_runtime_entry(entry)) continue;
        std::vector<uint8_t> bytes;
        if (!decompress_entry(bundle, entry, &bytes, error) ||
            !write_file(
                temporary / std::filesystem::u8path(entry.name),
                bytes,
                error) ||
            !cached_file_valid(temporary, entry)) {
            remove_generated_tree(temporary, parent);
            return false;
        }
    }

    const std::filesystem::path backup =
        parent / std::filesystem::path(cache.filename().wstring() + L".old");
    remove_generated_tree(backup, parent);
    const bool cache_existed = std::filesystem::exists(cache, ec) && !ec;
    if (cache_existed &&
        !MoveFileExW(cache.c_str(), backup.c_str(), MOVEFILE_WRITE_THROUGH)) {
        if (error != nullptr) *error = windows_error("Cannot replace damaged runtime cache");
        remove_generated_tree(temporary, parent);
        return false;
    }
    if (!MoveFileExW(temporary.c_str(), cache.c_str(), MOVEFILE_WRITE_THROUGH)) {
        const DWORD move_error = GetLastError();
        if (cache_existed) {
            MoveFileExW(backup.c_str(), cache.c_str(), MOVEFILE_WRITE_THROUGH);
        }
        remove_generated_tree(temporary, parent);
        if (error != nullptr) *error = windows_error("Cannot commit runtime cache", move_error);
        return false;
    }
    if (cache_existed) remove_generated_tree(backup, parent);
    return cache_baseline_valid(bundle, cache);
}

bool ensure_single_entry(
    const BundleView& bundle,
    const BundleEntry& entry,
    const std::filesystem::path& cache,
    std::string* error) {
    if (cached_file_valid(cache, entry)) return true;
    if (!enough_disk_space(cache, entry.uncompressed_size, error)) return false;

    std::vector<uint8_t> bytes;
    if (!decompress_entry(bundle, entry, &bytes, error)) return false;
    const std::filesystem::path target =
        cache / std::filesystem::u8path(entry.name);
    const std::filesystem::path temporary =
        cache / std::filesystem::path(
            std::filesystem::u8path(entry.name).wstring() +
            L".tmp." + std::to_wstring(GetCurrentProcessId()));
    std::error_code ec;
    std::filesystem::remove(temporary, ec);
    if (!write_file(temporary, bytes, error)) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    if (!MoveFileExW(
            temporary.c_str(),
            target.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD move_error = GetLastError();
        std::filesystem::remove(temporary, ec);
        if (error != nullptr) {
            *error = std::string("Failed to extract ") + entry.name + ": " +
                windows_error("atomic replace failed", move_error);
        }
        return false;
    }
    if (!cached_file_valid(cache, entry)) {
        if (error != nullptr) {
            *error = std::string("SHA-256 mismatch for ") + entry.name;
        }
        return false;
    }
    return true;
}

std::wstring mutex_name_for_hash(const std::string& hash) {
    const std::string short_hash = hash.substr(0, std::min<size_t>(16, hash.size()));
    return L"Local\\cq_ai_runtime_v23_5_" +
        std::wstring(short_hash.begin(), short_hash.end());
}

bool prepare_bundle_and_cache(
    BundleView* bundle,
    std::filesystem::path* cache,
    std::string* bundle_hash,
    std::string* error) {
    if (!get_resource_bundle(bundle, error)) return false;
    std::array<uint8_t, 32> digest{};
    if (!sha256_bytes(bundle->bytes, bundle->size, &digest)) {
        if (error != nullptr) *error = "Cannot calculate embedded runtime package SHA-256";
        return false;
    }
    *bundle_hash = hex_digest(digest.data(), digest.size());
    const std::filesystem::path local = local_appdata_path(error);
    if (local.empty()) return false;
    *cache = local / L"CQ_AI" / L"runtime" / L"v23.5" /
        std::filesystem::path(
            L"ort-dml-1.24.4-" +
            std::wstring(bundle_hash->begin(), bundle_hash->end()));
    return true;
}

} // namespace

bool initialize_embedded_runtime(std::string* error) {
    std::lock_guard<std::mutex> state_lock(g_state_mutex);
    if (g_initialized) return true;

    BundleView bundle;
    std::filesystem::path cache;
    std::string bundle_hash;
    if (!prepare_bundle_and_cache(
            &bundle, &cache, &bundle_hash, error)) {
        return false;
    }

    ScopedMutex cache_mutex;
    if (!cache_mutex.acquire(mutex_name_for_hash(bundle_hash), error) ||
        !build_baseline_cache(bundle, cache, error)) {
        return false;
    }

    if (!SetDefaultDllDirectories(
            LOAD_LIBRARY_SEARCH_SYSTEM32 |
            LOAD_LIBRARY_SEARCH_USER_DIRS)) {
        if (error != nullptr) *error = windows_error("SetDefaultDllDirectories failed");
        return false;
    }
    if (g_dll_directory_cookie == nullptr) {
        g_dll_directory_cookie = AddDllDirectory(cache.c_str());
        if (g_dll_directory_cookie == nullptr) {
            if (error != nullptr) *error = windows_error("AddDllDirectory failed");
            return false;
        }
    }

    const std::filesystem::path ort_path = cache / L"onnxruntime.dll";
    HMODULE ort_module = LoadLibraryExW(
        ort_path.c_str(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
            LOAD_LIBRARY_SEARCH_SYSTEM32 |
            LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (ort_module == nullptr) {
        const DWORD code = GetLastError();
        if (error != nullptr) {
            *error = windows_error(
                "Failed to load embedded onnxruntime.dll", code);
        }
        return false;
    }

    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    const auto get_api_base = reinterpret_cast<GetApiBaseFn>(
        GetProcAddress(ort_module, "OrtGetApiBase"));
    const OrtApiBase* api_base =
        get_api_base == nullptr ? nullptr : get_api_base();
    const OrtApi* api =
        api_base == nullptr ? nullptr : api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        FreeLibrary(ort_module);
        if (error != nullptr) {
            *error = "Embedded onnxruntime.dll does not expose the required ORT API";
        }
        return false;
    }
    Ort::InitApi(api);

    wchar_t loaded_path[32768]{};
    const DWORD loaded_length = GetModuleFileNameW(
        ort_module,
        loaded_path,
        static_cast<DWORD>(std::size(loaded_path)));
    g_loaded_ort_path = loaded_length > 0 &&
            loaded_length < std::size(loaded_path)
        ? wide_to_utf8(std::wstring(loaded_path, loaded_length))
        : wide_to_utf8(ort_path.wstring());
    g_ort_module = ort_module;
    g_cache_path = std::move(cache);
    g_bundle_hash = std::move(bundle_hash);
    g_directml_cached = false;
    if (const BundleEntry* dml = find_entry(bundle, "DirectML.dll")) {
        g_directml_cached = cached_file_valid(g_cache_path, *dml);
    }
    g_initialized = true;
    return true;
}

bool ensure_embedded_directml(std::string* error) {
    if (!initialize_embedded_runtime(error)) return false;
    std::lock_guard<std::mutex> state_lock(g_state_mutex);

    BundleView bundle;
    std::string bundle_error;
    if (!get_resource_bundle(&bundle, &bundle_error)) {
        if (error != nullptr) *error = bundle_error;
        return false;
    }
    const BundleEntry* entry = find_entry(bundle, "DirectML.dll");
    if (entry == nullptr ||
        (entry->flags & kEntryDirectML) == 0) {
        if (error != nullptr) *error = "DirectML.dll is missing from embedded runtime package";
        return false;
    }
    ScopedMutex cache_mutex;
    if (!cache_mutex.acquire(mutex_name_for_hash(g_bundle_hash), error) ||
        !ensure_single_entry(bundle, *entry, g_cache_path, error)) {
        return false;
    }
    g_directml_cached = true;
    return true;
}

bool verify_embedded_runtime(
    std::string* report,
    std::string* error) {
    BundleView bundle;
    if (!get_resource_bundle(&bundle, error)) return false;
    uint64_t raw_total = 0;
    std::string manifest_json;
    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        std::vector<uint8_t> bytes;
        if (!decompress_entry(bundle, bundle.entries[i], &bytes, error)) {
            return false;
        }
        raw_total += bytes.size();
        if (std::string(bundle.entries[i].name) == "runtime-manifest.json") {
            manifest_json.assign(
                reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
    }
    if (manifest_json.empty() || manifest_json.front() != '{') {
        if (error != nullptr) *error = "Embedded runtime manifest is missing or invalid";
        return false;
    }
    std::array<uint8_t, 32> digest{};
    if (!sha256_bytes(bundle.bytes, bundle.size, &digest)) {
        if (error != nullptr) *error = "Cannot calculate embedded runtime package SHA-256";
        return false;
    }
    if (report != nullptr) {
        std::ostringstream out;
        out << "{\"valid\":true,\"format_version\":"
            << bundle.header->format_version
            << ",\"file_count\":" << bundle.header->file_count
            << ",\"packed_size\":" << bundle.size
            << ",\"uncompressed_size\":" << raw_total
            << ",\"bundle_sha256\":\""
            << hex_digest(digest.data(), digest.size())
            << "\",\"manifest\":" << manifest_json << '}';
        *report = out.str();
    }
    return true;
}

bool embedded_third_party_notices(
    std::string* notices,
    std::string* error) {
    if (notices == nullptr) return false;
    BundleView bundle;
    if (!get_resource_bundle(&bundle, error)) return false;
    std::ostringstream out;
    for (uint32_t i = 0; i < bundle.header->file_count; ++i) {
        const BundleEntry& entry = bundle.entries[i];
        if ((entry.flags & kEntryNotice) == 0) continue;
        std::vector<uint8_t> bytes;
        if (!decompress_entry(bundle, entry, &bytes, error)) return false;
        out << "\n===== " << entry.name << " =====\n";
        out.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        if (bytes.empty() || bytes.back() != '\n') out << '\n';
    }
    *notices = out.str();
    return true;
}

std::string runtime_cache_path_utf8() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return wide_to_utf8(g_cache_path.wstring());
}

std::string runtime_bundle_hash() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_bundle_hash;
}

std::string loaded_ort_path_utf8() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_loaded_ort_path;
}

bool directml_is_cached() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_directml_cached;
}

} // namespace ai_runtime
