#pragma once
// Read only the ONNX descriptor, not weights or executable Python metadata.
// Wire fields follow the Apache-2.0 ONNX ModelProto/GraphProto specification.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ai {
enum class YoloFormat : int32_t { Dense = 0, Objectness = 1, EndToEnd = 2 };
enum class YoloLayout : int32_t { BCN = 0, BNC = 1 };
struct YoloTensor {
    std::string name;
    int32_t type = 0;
    std::vector<int64_t> shape;
};
struct YoloModelDescriptor {
    YoloFormat format = YoloFormat::Dense;
    YoloLayout layout = YoloLayout::BCN;
    int32_t classes = 0;
    YoloTensor input, output;
};
inline const char* yolo_format_name(YoloFormat f) {
    return f == YoloFormat::Dense ? "dense" : f == YoloFormat::Objectness ? "objectness" : "end2end";
}
inline const char* yolo_layout_name(YoloLayout f) { return f == YoloLayout::BCN ? "BCN" : "BNC"; }

namespace yolo_model_detail {
inline void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Field {
    uint32_t id = 0, wire = 0;
    uint64_t number = 0;
    std::string_view bytes;
};
class Reader {
    std::string_view bytes_;
    size_t position_ = 0, fields_ = 0;
    uint64_t varint() {
        uint64_t value = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            require(position_ < bytes_.size(), "Truncated ONNX varint");
            const auto c = static_cast<uint8_t>(bytes_[position_++]);
            require(shift < 63 || c <= 1, "Overflowing ONNX varint");
            value |= static_cast<uint64_t>(c & 127) << shift;
            if (!(c & 128)) return value;
        }
        throw std::runtime_error("Overflowing ONNX varint");
    }
public:
    explicit Reader(std::string_view bytes) : bytes_(bytes) {}
    bool next(Field& f) {
        if (position_ == bytes_.size()) return false;
        require(++fields_ <= 2000000, "ONNX descriptor field limit exceeded");
        f = {};
        const auto tag = varint();
        require((tag >> 3) > 0 && (tag >> 3) <= 0x1fffffff, "Invalid ONNX field tag");
        f.id = static_cast<uint32_t>(tag >> 3); f.wire = static_cast<uint32_t>(tag & 7);
        if (f.wire == 0) { f.number = varint(); return true; }
        require(f.wire == 1 || f.wire == 2 || f.wire == 5, "Unsupported ONNX wire type");
        const uint64_t count = f.wire == 2 ? varint() : f.wire == 1 ? 8 : 4;
        require(count <= bytes_.size() - position_, "Truncated ONNX field");
        f.bytes = bytes_.substr(position_, static_cast<size_t>(count));
        position_ += static_cast<size_t>(count);
        return true;
    }
};
inline std::string text(const Field& f) {
    require(f.wire == 2 && f.bytes.size() <= 1024 * 1024, "Invalid or oversized ONNX descriptor string");
    return std::string(f.bytes);
}
inline void message(const Field& f) { require(f.wire == 2, "Invalid ONNX descriptor message"); }
inline YoloTensor tensor(std::string_view bytes) {
    YoloTensor value; Reader r(bytes); Field f;
    while (r.next(f)) {
        if (f.id == 1) value.name = text(f);
        if (f.id != 2) continue;
        message(f); Reader type(f.bytes); Field t;
        while (type.next(t)) {
            if (t.id != 1) continue;
            message(t); Reader tt(t.bytes); Field a;
            while (tt.next(a)) {
                if (a.id == 1) { require(a.wire == 0 && a.number < 256, "Invalid ONNX tensor type"); value.type = static_cast<int32_t>(a.number); }
                if (a.id != 2) continue;
                message(a); Reader shape(a.bytes); Field d;
                while (shape.next(d)) {
                    if (d.id != 1) continue;
                    message(d); require(value.shape.size() < 16, "ONNX tensor rank limit exceeded");
                    int64_t dim = -1; Reader dimension(d.bytes); Field e; bool concrete = false;
                    while (dimension.next(e)) {
                        if (e.id == 1) {
                            require(e.wire == 0 && !concrete && e.number <= INT64_MAX, "Invalid ONNX dimension");
                            dim = static_cast<int64_t>(e.number); concrete = true;
                        }
                        if (e.id == 2) { require(!concrete, "Conflicting ONNX dimension"); (void)text(e); }
                    }
                    value.shape.push_back(dim);
                }
            }
        }
    }
    return value;
}
struct Model {
    std::map<std::string, std::string> metadata;
    std::vector<YoloTensor> inputs, outputs;
    bool nms = false;
};
inline void graph(std::string_view bytes, Model& m, int depth);
inline void node(std::string_view bytes, Model& m, int depth) {
    require(depth < 32, "ONNX nested graph limit exceeded");
    Reader r(bytes); Field f;
    while (r.next(f)) {
        if (f.id == 4) {
            auto op = text(f);
            std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (op.find("nonmaxsuppression") != std::string::npos || op.find("nms") != std::string::npos) m.nms = true;
        }
        if (f.id == 5) {
            message(f); Reader attributes(f.bytes); Field a;
            while (attributes.next(a)) if (a.id == 6 || a.id == 11) { message(a); graph(a.bytes, m, depth + 1); }
        }
    }
}
inline void graph(std::string_view bytes, Model& m, int depth) {
    require(depth < 32, "ONNX nested graph limit exceeded");
    Reader r(bytes); Field f;
    while (r.next(f)) {
        if (f.id == 1) { message(f); node(f.bytes, m, depth); }
        if (f.id == 5) {
            message(f); Reader weights(f.bytes); Field w;
            while (weights.next(w)) if (w.id == 14) require(w.wire == 0 && w.number == 0, "YOLO ONNX external weights must be consolidated into one model");
        }
        if (depth == 0 && (f.id == 11 || f.id == 12)) {
            message(f); auto& list = f.id == 11 ? m.inputs : m.outputs;
            require(list.size() < 128, "ONNX I/O count limit exceeded"); list.push_back(tensor(f.bytes));
        }
    }
}
inline Model read(const void* data, size_t size) {
    require(data && size && size <= 512ull * 1024 * 1024, "YOLO ONNX model size is invalid (limit 512 MiB)");
    Model m; Reader r(std::string_view(static_cast<const char*>(data), size)); Field f; bool seen_graph = false;
    while (r.next(f)) {
        if (f.id == 7) { message(f); require(!seen_graph, "Duplicate ONNX graph"); graph(f.bytes, m, 0); seen_graph = true; }
        if (f.id == 14) {
            message(f); Reader entry(f.bytes); Field p; std::string key, value;
            while (entry.next(p)) { if (p.id == 1) key = text(p); if (p.id == 2) value = text(p); }
            require(!key.empty() && m.metadata.size() < 1024 && m.metadata.emplace(key, value).second, "Invalid or duplicate ONNX metadata");
        }
        // FunctionProto.node = 7; inspect local functions too, including their subgraphs.
        if (f.id == 25) {
            message(f); Reader function(f.bytes); Field n;
            while (function.next(n)) if (n.id == 7) { message(n); node(n.bytes, m, 0); }
        }
    }
    require(seen_graph, "ONNX graph is missing"); return m;
}
inline void spaces(std::string_view s, size_t& p) { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p; }
inline void quoted(std::string_view s, size_t& p) {
    require(p < s.size() && (s[p] == '\'' || s[p] == '"'), "Invalid YOLO names metadata");
    const char quote = s[p++]; bool closed = false;
    while (p < s.size()) { const char c = s[p++]; if (c == '\\') { require(p < s.size(), "Truncated metadata escape"); ++p; } else if (c == quote) { closed = true; break; } }
    require(closed, "Unterminated metadata string");
}
inline int positive(std::string_view s) {
    require(!s.empty() && s.size() < 7, "Invalid YOLO class count");
    int count = 0;
    for (char c : s) { require(c >= '0' && c <= '9', "Invalid YOLO class count"); count = count * 10 + c - '0'; }
    require(count > 0 && count <= 65531, "YOLO class count outside 1..65531"); return count;
}
// Accept the simple integer-key/string-value dict or string list emitted by
// official exporters; never evaluate Python or count punctuation inside names.
inline int names_count(std::string_view s) {
    size_t p = 0; spaces(s, p); require(p < s.size() && (s[p] == '{' || s[p] == '['), "Invalid YOLO names metadata");
    const bool dictionary = s[p++] == '{'; const char end = dictionary ? '}' : ']'; int count = 0;
    for (;;) {
        spaces(s, p); require(p < s.size(), "Truncated YOLO names metadata");
        if (s[p] == end) { ++p; break; }
        require(count < 65531, "YOLO class count limit exceeded");
        if (dictionary) {
            const bool quoted_key = s[p] == '\'' || s[p] == '"'; const char q = s[p]; if (quoted_key) ++p;
            size_t begin = p; while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p;
            require(p > begin && p - begin <= 5, "YOLO names keys must be contiguous integers");
            int key = 0; for (size_t i = begin; i < p; ++i) key = key * 10 + s[i] - '0';
            require(key == count, "YOLO names keys must start at0 and be contiguous");
            if (quoted_key) { require(p < s.size() && s[p] == q, "Invalid metadata key"); ++p; }
            spaces(s, p); require(p < s.size() && s[p++] == ':', "Invalid metadata separator"); spaces(s, p);
        }
        quoted(s, p); ++count; spaces(s, p);
        require(p < s.size(), "Truncated YOLO names metadata");
        if (s[p] == ',') { ++p; continue; }
        require(s[p] == end, "Invalid YOLO names metadata");
    }
    spaces(s, p); require(p == s.size() && count > 0, "Invalid YOLO names metadata"); return count;
}
inline int boolean(const std::string& value) {
    if (value == "True" || value == "true" || value == "1") return 1;
    if (value == "False" || value == "false" || value == "0") return 0;
    throw std::runtime_error("Invalid YOLO end2end metadata");
}
inline bool candidates_match(int64_t count, const YoloTensor& input, bool anchors) {
    if (input.shape.size() != 4 || input.shape[2] <= 0 || input.shape[3] <= 0) return false;
    int64_t cells = 0;
    for (int stride : {8, 16, 32, 64}) {
        cells += (input.shape[2] / stride) * (input.shape[3] / stride);
        if (stride >= 32 && count == cells * (anchors ? 3 : 1)) return true;
    }
    return false;
}
inline YoloModelDescriptor resolve(const Model& m) {
    require(!m.nms, "Embedded NMS is unsupported; re-export the raw or native end2end detection head");
    require(m.inputs.size() == 1 && m.outputs.size() == 1, "YOLO requires exactly one input and one decoded detection output");
    const auto get = [&](const char* key) { auto it = m.metadata.find(key); return it == m.metadata.end() ? std::string{} : it->second; };
    const auto task = get("task"); require(task.empty() || task == "detect", "Only YOLO task=detect is supported (no segment/pose/OBB/classify)");
    YoloModelDescriptor d; d.input = m.inputs[0]; d.output = m.outputs[0];
    require(d.input.type == 1 && d.output.type == 1, "YOLO model input/output must be FP32");
    require(d.input.shape.size() == 4 && (d.input.shape[0] == 1 || d.input.shape[0] == -1) && d.input.shape[1] == 3,
            "YOLO input must be NCHW with batch1 and three channels");
    require(d.input.shape[2] != 0 && d.input.shape[3] != 0 && d.input.shape[2] <= 4096 && d.input.shape[3] <= 4096,
            "YOLO input dimensions are invalid (limit4096)");
    require(d.input.shape[2] < 0 || d.input.shape[3] < 0 || d.input.shape[2] == d.input.shape[3], "YOLO input must be square");
    require(d.output.shape.size() == 3 && (d.output.shape[0] == 1 || d.output.shape[0] == -1), "YOLO output must be a batch1 rank3 tensor");
    require(d.output.shape[1] >= -1 && d.output.shape[2] >= -1 && d.output.shape[1] <= 1000000 && d.output.shape[2] <= 1000000,
            "YOLO output dimensions exceed limits");
    if (d.output.shape[1] >= 0 && d.output.shape[2] >= 0)
        require(static_cast<uint64_t>(d.output.shape[1]) * d.output.shape[2] <= 512ull * 1024 * 1024 / sizeof(float), "YOLO output exceeds512 MiB");
    int nc = 0; const auto names = get("names"), classes = get("cq.yolo.classes");
    if (!names.empty()) nc = names_count(names);
    if (!classes.empty()) { const int n = positive(classes); require(!nc || nc == n, "Conflicting YOLO class metadata"); nc = n; }
    const auto schema = get("cq.yolo.schema"); require(schema.empty() || schema == "1", "Unsupported CQ YOLO metadata schema");
    const auto format = get("cq.yolo.format"), layout = get("cq.yolo.layout"), end = get("end2end");
    require(format.empty() || format == "dense" || format == "objectness" || format == "end2end", "Unknown CQ YOLO format");
    require(layout.empty() || layout == "BCN" || layout == "BNC", "Unknown CQ YOLO layout");
    const int end2end = end.empty() ? -1 : boolean(end);
    if (!format.empty() && end2end >= 0) require((format == "end2end") == (end2end == 1), "Conflicting YOLO end2end metadata");
    if (format == "end2end" || end2end == 1) {
        require(nc > 0 && d.output.shape[2] == 6 && (layout.empty() || layout == "BNC"), "Native end2end requires known classes and output [1,K,6]");
        d.format = YoloFormat::EndToEnd; d.layout = YoloLayout::BNC; d.classes = nc; return d;
    }
    std::vector<YoloModelDescriptor> matches;
    require(!layout.empty() || d.output.shape[1] != d.output.shape[2], "Ambiguous YOLO output axes; supply explicit layout metadata");
    for (auto f : {YoloFormat::Dense, YoloFormat::Objectness}) for (auto l : {YoloLayout::BCN, YoloLayout::BNC}) {
        if (!format.empty() && format != yolo_format_name(f)) continue;
        if (!layout.empty() && layout != yolo_layout_name(l)) continue;
        const int attr_axis = l == YoloLayout::BCN ? 1 : 2, candidate_axis = 3 - attr_axis;
        const int64_t attrs = d.output.shape[attr_axis], count = d.output.shape[candidate_axis];
        const int extra = f == YoloFormat::Dense ? 4 : 5;
        if (attrs <= extra || attrs - extra > 65531 || (nc && attrs != nc + extra)) continue;
        const bool canonical = candidates_match(count, d.input, f == YoloFormat::Objectness);
        if (format.empty() && !nc && !canonical && !(f == YoloFormat::Dense && l == YoloLayout::BCN && count == -1)) continue;
        // A six-column tensor could also be an end-to-end result. Names alone
        // cannot distinguish it from a one-class v5/custom export.
        if (format.empty() && l == YoloLayout::BNC && attrs == 6 && end2end < 0) continue;
        auto candidate = d; candidate.format = f; candidate.layout = l; candidate.classes = static_cast<int32_t>(attrs - extra); matches.push_back(candidate);
    }
    require(matches.size() == 1, "Ambiguous or unsupported YOLO output; inspect and prepare a metadata-only copy with explicit format/classes/layout");
    return matches[0];
}
} // namespace yolo_model_detail

inline bool inspect_yolo_model(const void* data, size_t size, YoloModelDescriptor* descriptor, std::string* error) {
    try { *descriptor = yolo_model_detail::resolve(yolo_model_detail::read(data, size)); return true; }
    catch (const std::exception& e) { if (error) *error = e.what(); return false; }
}
inline bool inspect_yolo_file(const char* path, YoloModelDescriptor* descriptor, std::string* error) {
    try {
        const auto file = std::filesystem::u8path(path);
        const auto size = std::filesystem::file_size(file);
        yolo_model_detail::require(size && size <= 512ull * 1024 * 1024, "Invalid YOLO file size");
        std::ifstream stream(file, std::ios::binary); std::vector<char> data(static_cast<size_t>(size));
        yolo_model_detail::require(static_cast<bool>(stream.read(data.data(), static_cast<std::streamsize>(size))), "Cannot read YOLO model");
        return inspect_yolo_model(data.data(), data.size(), descriptor, error);
    } catch (const std::exception& e) { if (error) *error = e.what(); return false; }
}
inline bool validate_yolo_output(const YoloModelDescriptor& d, int64_t batch, int64_t axis1, int64_t axis2, size_t elements) {
    if (batch != 1 || axis1 < 0 || axis2 < 0 || axis1 > 1000000 || axis2 > 1000000) return false;
    const uint64_t product = static_cast<uint64_t>(axis1) * axis2;
    if (product != elements || product > 512ull * 1024 * 1024 / sizeof(float)) return false;
    if (d.output.shape.size() != 3) return false;
    if ((d.output.shape[1] >= 0 && d.output.shape[1] != axis1) || (d.output.shape[2] >= 0 && d.output.shape[2] != axis2)) return false;
    const int64_t attrs = d.layout == YoloLayout::BCN ? axis1 : axis2;
    return attrs == (d.format == YoloFormat::EndToEnd ? 6 : d.classes + (d.format == YoloFormat::Objectness ? 5 : 4));
}
} // namespace ai
