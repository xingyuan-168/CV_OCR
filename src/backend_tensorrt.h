#pragma once
#include "backends.h"
namespace ai {
std::unique_ptr<YoloBackend> create_tensorrt_yolo_backend(const Config&, std::string*);
}
