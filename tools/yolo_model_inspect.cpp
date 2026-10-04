#include "yolo_model.h"
#include <iostream>
#include <iomanip>
#include <sstream>
namespace {
std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
void tensor(const ai::YoloTensor& t) {
    std::cout << "{\"name\":" << quote(t.name) << ",\"dtype\":\"float32\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) { if (i) std::cout << ','; std::cout << t.shape[i]; }
    std::cout << "]}";
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    if (argc != 2) { std::cerr << "Usage: yolo_model_inspect MODEL.onnx\n"; return 2; }
    const auto path = std::filesystem::path(argv[1]).u8string();
    ai::YoloModelDescriptor d; std::string error;
    if (!ai::inspect_yolo_file(path.c_str(), &d, &error)) {
        std::cout << "{\"compatible\":false,\"error\":" << quote(error) << "}\n"; return 2;
    }
    std::cout << "{\"compatible\":true,\"descriptor_schema\":1,\"format\":" << quote(ai::yolo_format_name(d.format))
        << ",\"layout\":" << quote(ai::yolo_layout_name(d.layout)) << ",\"classes\":" << d.classes
        << ",\"coordinates\":" << quote(d.format == ai::YoloFormat::EndToEnd ? "xyxy" : "cxcywh")
        << ",\"external_nms\":" << (d.format == ai::YoloFormat::EndToEnd ? "false" : "true") << ",\"input\":";
    tensor(d.input); std::cout << ",\"output\":"; tensor(d.output); std::cout << "}\n"; return 0;
}
