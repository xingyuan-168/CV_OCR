#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "ai_engine.h"
#include "compact_result_test_utils.h"
#include "coordinate_offset.h"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <windows.h>

namespace {

struct ZipSource {
    std::string name;
    std::vector<uint8_t> data;
    bool deflate = false;
    bool utf8 = true;
};

void append_u16(std::vector<uint8_t>* output, uint16_t value) {
    output->push_back(static_cast<uint8_t>(value));
    output->push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>* output, uint32_t value) {
    output->push_back(static_cast<uint8_t>(value));
    output->push_back(static_cast<uint8_t>(value >> 8));
    output->push_back(static_cast<uint8_t>(value >> 16));
    output->push_back(static_cast<uint8_t>(value >> 24));
}

void set_u16(std::vector<uint8_t>* output, size_t offset, uint16_t value) {
    assert(output != nullptr && offset + 2 <= output->size());
    (*output)[offset] = static_cast<uint8_t>(value);
    (*output)[offset + 1] = static_cast<uint8_t>(value >> 8);
}

void set_u32(std::vector<uint8_t>* output, size_t offset, uint32_t value) {
    assert(output != nullptr && offset + 4 <= output->size());
    (*output)[offset] = static_cast<uint8_t>(value);
    (*output)[offset + 1] = static_cast<uint8_t>(value >> 8);
    (*output)[offset + 2] = static_cast<uint8_t>(value >> 16);
    (*output)[offset + 3] = static_cast<uint8_t>(value >> 24);
}

uint32_t get_u32(const std::vector<uint8_t>& input, size_t offset) {
    assert(offset + 4 <= input.size());
    return static_cast<uint32_t>(input[offset]) |
        (static_cast<uint32_t>(input[offset + 1]) << 8) |
        (static_cast<uint32_t>(input[offset + 2]) << 16) |
        (static_cast<uint32_t>(input[offset + 3]) << 24);
}

uint32_t crc32_bytes(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xffffffffu;
    for (uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

std::string wide_to_acp(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_ACP, 0, text, -1, nullptr, 0, nullptr, nullptr);
    assert(size > 1);
    std::string output(static_cast<size_t>(size), '\0');
    assert(WideCharToMultiByte(
        CP_ACP, 0, text, -1, output.data(), size, nullptr, nullptr) == size);
    output.pop_back();
    return output;
}

std::vector<uint8_t> raw_stored_deflate(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> output;
    size_t offset = 0;
    do {
        const size_t remaining = data.size() - offset;
        const uint16_t block_size = static_cast<uint16_t>(
            std::min<size_t>(remaining, std::numeric_limits<uint16_t>::max()));
        const bool final_block = offset + block_size == data.size();
        output.push_back(final_block ? 0x01 : 0x00);
        append_u16(&output, block_size);
        append_u16(&output, static_cast<uint16_t>(~block_size));
        output.insert(
            output.end(),
            data.begin() + static_cast<std::ptrdiff_t>(offset),
            data.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
        offset += block_size;
    } while (offset < data.size());
    return output;
}

std::vector<uint8_t> make_zip(const std::vector<ZipSource>& sources) {
    struct CentralEntry {
        ZipSource source;
        std::vector<uint8_t> compressed;
        uint32_t crc = 0;
        uint32_t local_offset = 0;
    };
    std::vector<uint8_t> output;
    std::vector<CentralEntry> entries;
    for (const ZipSource& source : sources) {
        CentralEntry entry;
        entry.source = source;
        entry.compressed = source.deflate ? raw_stored_deflate(source.data) : source.data;
        entry.crc = crc32_bytes(source.data);
        entry.local_offset = static_cast<uint32_t>(output.size());
        const uint16_t flags = source.utf8 ? static_cast<uint16_t>(1u << 11) : 0;
        append_u32(&output, 0x04034b50u);
        append_u16(&output, 20);
        append_u16(&output, flags);
        append_u16(&output, source.deflate ? 8 : 0);
        append_u16(&output, 0);
        append_u16(&output, 0);
        append_u32(&output, entry.crc);
        append_u32(&output, static_cast<uint32_t>(entry.compressed.size()));
        append_u32(&output, static_cast<uint32_t>(source.data.size()));
        append_u16(&output, static_cast<uint16_t>(source.name.size()));
        append_u16(&output, 0);
        output.insert(output.end(), source.name.begin(), source.name.end());
        output.insert(output.end(), entry.compressed.begin(), entry.compressed.end());
        entries.push_back(std::move(entry));
    }

    const uint32_t central_offset = static_cast<uint32_t>(output.size());
    for (const CentralEntry& entry : entries) {
        const uint16_t flags = entry.source.utf8 ? static_cast<uint16_t>(1u << 11) : 0;
        append_u32(&output, 0x02014b50u);
        append_u16(&output, 20);
        append_u16(&output, 20);
        append_u16(&output, flags);
        append_u16(&output, entry.source.deflate ? 8 : 0);
        append_u16(&output, 0);
        append_u16(&output, 0);
        append_u32(&output, entry.crc);
        append_u32(&output, static_cast<uint32_t>(entry.compressed.size()));
        append_u32(&output, static_cast<uint32_t>(entry.source.data.size()));
        append_u16(&output, static_cast<uint16_t>(entry.source.name.size()));
        append_u16(&output, 0);
        append_u16(&output, 0);
        append_u16(&output, 0);
        append_u16(&output, 0);
        append_u32(&output, 0);
        append_u32(&output, entry.local_offset);
        output.insert(output.end(), entry.source.name.begin(), entry.source.name.end());
    }
    const uint32_t central_size = static_cast<uint32_t>(output.size()) - central_offset;
    append_u32(&output, 0x06054b50u);
    append_u16(&output, 0);
    append_u16(&output, 0);
    append_u16(&output, static_cast<uint16_t>(entries.size()));
    append_u16(&output, static_cast<uint16_t>(entries.size()));
    append_u32(&output, central_size);
    append_u32(&output, central_offset);
    append_u16(&output, 0);
    return output;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    assert(size > 0);
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    assert(input.good());
    return bytes;
}

size_t find_signature(const std::vector<uint8_t>& bytes, uint32_t signature, size_t begin = 0) {
    for (size_t i = begin; i + 4 <= bytes.size(); ++i) {
        const uint32_t value = static_cast<uint32_t>(bytes[i]) |
            (static_cast<uint32_t>(bytes[i + 1]) << 8) |
            (static_cast<uint32_t>(bytes[i + 2]) << 16) |
            (static_cast<uint32_t>(bytes[i + 3]) << 24);
        if (value == signature) return i;
    }
    return std::string::npos;
}

void test_coordinate_helpers() {
    CVMatchResult cv{10, 20, 30, 40, 0.9f, 2};
    assert(ai::coordinate::offset_cv_result(&cv, 100, -5));
    assert(cv.x == 110 && cv.y == 15 && cv.w == 30 && cv.h == 40);

    OCRTextResult text{25, 35, 10, 20, 30, 30, 0.8f};
    assert(ai::coordinate::offset_ocr_text_result(&text, -3, 7));
    assert(text.x == 7 && text.y == 27 && text.cx == 22 && text.cy == 42);

    OCRCoordResult coord{25, 35, 30, 30, 4};
    assert(ai::coordinate::offset_ocr_coord_result(&coord, 5, -10));
    assert(coord.x == 30 && coord.y == 25 && coord.target_index == 4);

    std::vector<AIDetectBox> boxes(1);
    boxes[0].x1 = 1.0f;
    boxes[0].y1 = 2.0f;
    boxes[0].x2 = 5.0f;
    boxes[0].y2 = 8.0f;
    assert(ai::coordinate::offset_yolo_boxes(&boxes, 10, 20));
    assert(std::fabs(boxes[0].x1 - 11.0f) < 0.001f);
    assert(std::fabs(boxes[0].y2 - 28.0f) < 0.001f);
    std::vector<AIDetectBox> yolo_overflow(1);
    yolo_overflow[0].x1 = 0.0f;
    yolo_overflow[0].x2 = 0.0f;
    assert(!ai::coordinate::offset_yolo_boxes(
        &yolo_overflow, std::numeric_limits<int32_t>::max(), 0));

    CVMatchResult overflow{1, 2, 3, 4, 0.0f, 0};
    const CVMatchResult original = overflow;
    assert(!ai::coordinate::offset_cv_result(
        &overflow, std::numeric_limits<int32_t>::max(), 0));
    assert(std::memcmp(&overflow, &original, sizeof(overflow)) == 0);
}

void test_zip_loader_and_cv_offsets() {
    const std::vector<uint8_t> templ = read_bytes(std::filesystem::u8path(u8"tests/fixtures/cv/base/电.bmp"));
    const std::vector<uint8_t> big = read_bytes(std::filesystem::u8path(u8"tests/fixtures/cv/base/大图1.bmp"));
    const std::vector<uint8_t> stored_zip = make_zip({
        ZipSource{u8"nested/电.bmp", templ, false, true},
        ZipSource{"notes/readme.txt", {'o', 'k'}, false, true},
        ZipSource{"empty-directory/", {}, false, true},
    });
    const std::vector<uint8_t> deflate_zip = make_zip({
        ZipSource{u8"nested/电.bmp", templ, true, true},
    });
    const std::string acp_template_name = wide_to_acp(L"电.bmp");
    const std::vector<uint8_t> acp_zip = make_zip({
        ZipSource{"legacy/" + acp_template_name, templ, false, false},
    });

    int32_t handle = 0;
    assert(CV_Create(&handle) == AI_OK && handle > 0);
    assert(CV_LoadTemplateZipFromMemory(
        handle, stored_zip.data(), static_cast<int32_t>(stored_zip.size())) == 1);

    CVMatchResult baseline{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &baseline, 0, 0) == 1);
    CVMatchResult shifted{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &shifted, 123, -45) == 1);
    assert(shifted.x == baseline.x + 123 && shifted.y == baseline.y - 45);
    assert(shifted.w == baseline.w && shifted.h == baseline.h && shifted.sim == baseline.sim);

    CVMatchResult transparent_baseline{};
    CVMatchResult transparent_shifted{};
    assert(CV_FindTransparentOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, "FF00FF", &transparent_baseline, 0, 0) == 1);
    assert(CV_FindTransparentOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, "FF00FF", &transparent_shifted, -73, 29) == 1);
    assert(transparent_shifted.x == transparent_baseline.x - 73);
    assert(transparent_shifted.y == transparent_baseline.y + 29);

    const std::string multi_text = CV_FindMultiText(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, 0, 123, -45);
    assert(compact_test::contains(multi_text, 0, baseline.x + 123, baseline.y - 45));
    const std::string transparent_text = CV_FindTransparentMultiText(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        "", 0.4f, "FF00FF", -73, 29);
    assert(compact_test::contains(
        transparent_text, 0, transparent_baseline.x - 73, transparent_baseline.y + 29));

    assert(CV_LoadTemplateZipFromMemory(
        handle, deflate_zip.data(), static_cast<int32_t>(deflate_zip.size())) == 1);
    CVMatchResult after_deflate{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &after_deflate, 0, 0) == 1);
    assert(CV_LoadTemplateZipFromMemory(
        handle, acp_zip.data(), static_cast<int32_t>(acp_zip.size())) == 1);
    CVMatchResult after_acp{};
    assert(CV_FindOne(
        handle, acp_template_name.c_str(), big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &after_acp, 0, 0) == 1);

    std::vector<uint8_t> corrupt = deflate_zip;
    const size_t central = find_signature(corrupt, 0x02014b50u);
    assert(central != std::string::npos);
    corrupt[central + 16] ^= 0x80;
    assert(CV_LoadTemplateZipFromMemory(
        handle, corrupt.data(), static_cast<int32_t>(corrupt.size())) < 0);
    CVMatchResult after_failure{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &after_failure, 0, 0) == 1);

    std::vector<uint8_t> encrypted = stored_zip;
    const size_t encrypted_central = find_signature(encrypted, 0x02014b50u);
    assert(encrypted_central != std::string::npos);
    set_u16(&encrypted, 6, static_cast<uint16_t>(1u << 11 | 1u));
    set_u16(&encrypted, encrypted_central + 8, static_cast<uint16_t>(1u << 11 | 1u));
    assert(CV_LoadTemplateZipFromMemory(
        handle, encrypted.data(), static_cast<int32_t>(encrypted.size())) < 0);

    const std::vector<uint8_t> traversal_zip = make_zip({
        ZipSource{u8"../电.bmp", templ, false, true},
    });
    assert(CV_LoadTemplateZipFromMemory(
        handle, traversal_zip.data(), static_cast<int32_t>(traversal_zip.size())) < 0);

    std::vector<uint8_t> zip64 = stored_zip;
    const size_t zip64_central = find_signature(zip64, 0x02014b50u);
    set_u32(&zip64, zip64_central + 20, std::numeric_limits<uint32_t>::max());
    assert(CV_LoadTemplateZipFromMemory(
        handle, zip64.data(), static_cast<int32_t>(zip64.size())) < 0);

    std::vector<uint8_t> ratio_bomb = deflate_zip;
    const size_t bomb_central = find_signature(ratio_bomb, 0x02014b50u);
    const uint32_t compressed_size = get_u32(ratio_bomb, bomb_central + 20);
    assert(compressed_size > 0 && compressed_size < 64u * 1024u * 1024u / 201u);
    set_u32(&ratio_bomb, bomb_central + 24, compressed_size * 201u);
    assert(CV_LoadTemplateZipFromMemory(
        handle, ratio_bomb.data(), static_cast<int32_t>(ratio_bomb.size())) < 0);

    std::vector<uint8_t> ignored_method = stored_zip;
    const size_t first_central = find_signature(ignored_method, 0x02014b50u);
    const size_t second_central = find_signature(ignored_method, 0x02014b50u, first_central + 4);
    assert(second_central != std::string::npos);
    set_u16(&ignored_method, second_central + 10, 99);
    assert(CV_LoadTemplateZipFromMemory(
        handle, ignored_method.data(), static_cast<int32_t>(ignored_method.size())) == 1);

    const std::vector<uint8_t> duplicate_zip = make_zip({
        ZipSource{u8"a/电.bmp", templ, false, true},
        ZipSource{u8"b/电.bmp", templ, false, true},
    });
    assert(CV_LoadTemplateZipFromMemory(
        handle, duplicate_zip.data(), static_cast<int32_t>(duplicate_zip.size())) < 0);
    const std::vector<uint8_t> no_bmp_zip = make_zip({
        ZipSource{"readme.txt", {'n', 'o'}, false, true},
    });
    assert(CV_LoadTemplateZipFromMemory(
        handle, no_bmp_zip.data(), static_cast<int32_t>(no_bmp_zip.size())) < 0);
    const std::vector<uint8_t> invalid_bmp_zip = make_zip({
        ZipSource{"broken.bmp", {'B', 'M', 0, 0}, false, true},
    });
    assert(CV_LoadTemplateZipFromMemory(
        handle, invalid_bmp_zip.data(), static_cast<int32_t>(invalid_bmp_zip.size())) < 0);

    int32_t isolated_handle = 0;
    assert(CV_Create(&isolated_handle) == AI_OK && isolated_handle != handle);
    assert(CV_LoadTemplateZipFromMemory(
        isolated_handle, stored_zip.data(), static_cast<int32_t>(stored_zip.size())) == 1);
    assert(CV_ClearTemplateCache(isolated_handle) == AI_OK);
    CVMatchResult still_isolated{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &still_isolated, 0, 0) == 1);
    assert(CV_Release(isolated_handle) == AI_OK);

    std::atomic<bool> stop_search{false};
    std::atomic<int32_t> search_failures{0};
    std::thread searcher([&]() {
        while (!stop_search.load(std::memory_order_relaxed)) {
            CVMatchResult concurrent_result{};
            if (CV_FindOne(
                    handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
                    0.4f, 0, &concurrent_result, 0, 0) != 1) {
                ++search_failures;
            }
        }
    });
    for (int i = 0; i < 12; ++i) {
        const std::vector<uint8_t>& archive = (i % 2 == 0) ? stored_zip : deflate_zip;
        assert(CV_LoadTemplateZipFromMemory(
            handle, archive.data(), static_cast<int32_t>(archive.size())) == 1);
    }
    stop_search.store(true, std::memory_order_relaxed);
    searcher.join();
    assert(search_failures.load() == 0);

    std::vector<uint8_t> changed = templ;
    const uint32_t pixel_offset = static_cast<uint32_t>(changed[10]) |
        (static_cast<uint32_t>(changed[11]) << 8) |
        (static_cast<uint32_t>(changed[12]) << 16) |
        (static_cast<uint32_t>(changed[13]) << 24);
    assert(pixel_offset < changed.size());
    for (size_t i = pixel_offset; i < changed.size(); ++i) changed[i] ^= 0xff;
    CVMatchResult no_match{99, 98, 97, 96, 0.5f, 3};
    assert(CV_FindOne(
        handle, u8"电.bmp", changed.data(), static_cast<int32_t>(changed.size()),
        1.0f, 0, &no_match, 500, 600) == 0);
    const CVMatchResult zero{};
    assert(std::memcmp(&no_match, &zero, sizeof(no_match)) == 0);
    assert(std::strcmp(CV_FindMultiText(
        handle, u8"电.bmp", changed.data(), static_cast<int32_t>(changed.size()),
        "", 1.0f, 0, 500, 600), "") == 0);
    assert(AI_GetLastError()[0] == '\0');
    CVMatchResult no_transparent_match{99, 98, 97, 96, 0.5f, 3};
    assert(CV_FindTransparentOne(
        handle, u8"电.bmp", changed.data(), static_cast<int32_t>(changed.size()),
        1.0f, 0, "FF00FF", &no_transparent_match, 500, 600) == 0);
    assert(std::memcmp(&no_transparent_match, &zero, sizeof(no_transparent_match)) == 0);
    assert(std::strcmp(CV_FindTransparentMultiText(
        handle, u8"电.bmp", changed.data(), static_cast<int32_t>(changed.size()),
        "", 1.0f, "FF00FF", 500, 600), "") == 0);
    assert(AI_GetLastError()[0] == '\0');

    CVMatchResult overflow_result{};
    assert(CV_FindOne(
        handle, u8"电.bmp", big.data(), static_cast<int32_t>(big.size()),
        0.4f, 0, &overflow_result, std::numeric_limits<int32_t>::max(), 0) ==
        AI_ERR_INVALID_ARGUMENT);
    assert(std::memcmp(&overflow_result, &zero, sizeof(overflow_result)) == 0);
    assert(CV_Release(handle) == AI_OK);
}

} // namespace

int main() {
    test_coordinate_helpers();
    test_zip_loader_and_cv_offsets();
    return 0;
}
