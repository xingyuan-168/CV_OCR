#pragma once

#include <stdint.h>
#include "yolo_diagnostics.h"

namespace ai_worker {

static constexpr uint32_t kMagic = 0x4149574bu; // "KWIA" little-endian marker.
static constexpr uint32_t kVersion = 26;
static constexpr uint32_t kMaxPayload = 512u * 1024u * 1024u;

enum class RuntimeFlavor : uint32_t {
    Core = 0
};

constexpr const char* runtime_flavor_name(RuntimeFlavor) {
    return "core";
}

constexpr const char* pipe_name(RuntimeFlavor) {
    return R"(\\.\pipe\cq_ai_worker_v26_core_0146)";
}

constexpr const char* singleton_name(RuntimeFlavor) {
    return "Local\\cq_ai_worker_v26_core_0146_singleton";
}

static constexpr RuntimeFlavor kBuildFlavor = RuntimeFlavor::Core;

enum Command : uint32_t {
    CMD_OCR_LOAD_PATH = 1,
    CMD_OCR_LOAD_MEMORY = 2,
    CMD_OCR_RECOGNIZE = 3,
    CMD_OCR_FIND_ONE = 4,
    CMD_OCR_FIND_MULTI = 5,
    CMD_YOLO_LOAD_PATH = 6,
    CMD_YOLO_LOAD_MEMORY = 7,
    CMD_YOLO_INFER_JSON = 8,
    CMD_RELEASE = 9,
    CMD_SHUTDOWN = 10,
    CMD_OCR_FIND_ONE_COORD = 11,
    CMD_OCR_RELEASE = 14,
    CMD_YOLO_RELEASE = 15,
    CMD_RUNTIME_STATUS = 16,
    CMD_GET_LAST_LATENCY = 17,
    CMD_OCR_LOAD_MEMORY_EX = 18,
    CMD_GET_OCR_STAGE_LATENCY = 19,
    CMD_YOLO_CREATE = 20,
    CMD_YOLO_RUNTIME_STATUS = 21,
    CMD_YOLO_LAST_LATENCY = 22,
    CMD_OCR_LOAD_EMBEDDED = 23
};

#pragma pack(push, 1)
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t command;
    uint32_t payload_size;
    uint64_t request_id;
};

struct ResponseHeader {
    uint32_t magic;
    uint32_t version;
    int32_t status;
    uint32_t payload_size;
    uint64_t request_id;
    ai::YoloTiming timing;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 24, "v26 request header layout");
static_assert(sizeof(ResponseHeader) == 136, "v26 response header layout");

} // namespace ai_worker
