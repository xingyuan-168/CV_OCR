#include "yolo_decode.h"
#include <cassert>
#include <iostream>
using namespace ai;
using namespace ai::yolo_model_detail;

static YoloTensor decl(std::vector<int64_t> shape) { return {"tensor", 1, std::move(shape)}; }
static Model model(std::vector<int64_t> shape, std::map<std::string, std::string> meta = {}) {
    Model m; m.inputs = {decl({1,3,320,320})}; m.outputs = {decl(std::move(shape))}; m.metadata = std::move(meta); return m;
}
static bool rejected(const Model& m) { try { resolve(m); return false; } catch (const std::exception&) { return true; } }
static std::string var(uint64_t x) { std::string s; do { unsigned char c = x & 127; x >>= 7; s += static_cast<char>(c | (x ? 128 : 0)); } while (x); return s; }
static std::string integer(int tag, uint64_t n) { return var(static_cast<uint64_t>(tag) * 8) + var(n); }
static std::string bytes(int tag, const std::string& b) { return var(static_cast<uint64_t>(tag) * 8 + 2) + var(b.size()) + b; }
static std::string value_info(const char* name, const std::vector<int>& shape) {
    std::string dims; for (int n : shape) dims += bytes(1, integer(1, n));
    return bytes(1, name) + bytes(2, bytes(1, integer(1, 1) + bytes(2, dims)));
}
static int decode(const YoloModelDescriptor& d, const std::vector<float>& v, int64_t a, int64_t b, std::vector<AIDetectBox>& boxes, float threshold = .5f) {
    AIImage image{}; image.width = image.height = 320;
    std::vector<RawDetection> candidates;
    return yolo_decode(v.data(), d, a, b, v.size(), image, {1,0,0}, threshold, .45f, {}, candidates, &boxes);
}
int main() {
#ifdef _WIN32
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    const auto dense = resolve(model({1,6,2100}, {{"names", "{0:'a,b', 1:'c\\\'d'}"}}));
    assert(dense.classes == 2 && dense.format == YoloFormat::Dense && dense.layout == YoloLayout::BCN);
    assert(resolve(model({1,10,2100})).classes == 6); // metadata-free historical export
    const auto v5 = resolve(model({1,6300,7}, {{"names", "['a','b']"}}));
    assert(v5.format == YoloFormat::Objectness && v5.layout == YoloLayout::BNC);
    assert(rejected(model({1,6300,6}, {{"names", "{0:'one'}"}})));
    assert(resolve(model({1,6300,6}, {{"names", "{0:'one'}"},{"cq.yolo.format","objectness"}})).classes == 1);
    assert(resolve(model({1,7,6300}, {{"names", "{0:'a',1:'b'}"}})).layout == YoloLayout::BCN);
    assert(resolve(model({1,2100,6}, {{"cq.yolo.format","dense"},{"cq.yolo.layout","BNC"},{"cq.yolo.classes","2"}})).format == YoloFormat::Dense);
    assert(rejected(model({1,300,6}, {{"names", "{0:'one'}"}})));
    assert(rejected(model({1,6,6}, {{"names", "{0:'a',1:'b'}"}})));
    auto end = resolve(model({1,300,6}, {{"end2end","True"},{"names","{0:'person',1:'bus'}"}}));
    assert(end.format == YoloFormat::EndToEnd);
    assert(rejected(model({1,300,6}, {{"end2end","True"},{"cq.yolo.format","dense"},{"cq.yolo.classes","2"}})));
    assert(rejected(model({1,6,2100}, {{"task","segment"}})));
    assert(rejected(model({2,6,2100})));
    auto invalid = model({1,6,2100}); invalid.nms = true; assert(rejected(invalid));
    invalid = model({1,6,2100}); invalid.inputs[0].type = 10; assert(rejected(invalid));
    invalid = model({1,6,2100}); invalid.outputs.push_back(invalid.outputs[0]); assert(rejected(invalid));
    invalid = model({1,6,2100}); invalid.inputs[0].shape[3] = 640; assert(rejected(invalid));
    assert(rejected(model({1,6,2100}, {{"names","{1:'bad'}"}})));
    assert(rejected(model({1,6,2100}, {{"names","__import__('os')"}})));
    auto dynamic = model({-1,6,-1}, {{"names","['a','b']"}}); dynamic.inputs[0].shape = {-1,3,-1,-1};
    assert(resolve(dynamic).classes == 2);

    std::vector<AIDetectBox> boxes;
    auto obj = v5; obj.output.shape = {1,3,7};
    // A class confidence of .99 does not pass when objectness is .1.
    const std::vector<float> rows = {10,10,4,4,.1f,.99f,.01f, 20,20,4,4,1,.5f,.1f, 30,30,4,4,.8f,.2f,.75f};
    assert(decode(obj, rows, 3, 7, boxes) == 2);
    assert(boxes[0].class_id == 1 && boxes[0].score == .8f * .75f && boxes[0].x1 == 28);
    assert(boxes[1].score == .5f && boxes[1].x1 == 18); // equality accepted
    std::vector<float> transposed(rows.size()); for (int i=0;i<3;++i) for (int a=0;a<7;++a) transposed[a*3+i]=rows[i*7+a];
    auto obj_t = obj; obj_t.layout = YoloLayout::BCN; obj_t.output.shape = {1,7,3};
    assert(decode(obj_t, transposed, 7, 3, boxes) == 2 && boxes[0].class_id == 1);
    end.output.shape = {1,2,6};
    const std::vector<float> overlap = {10,10,30,30,.8f,0, 10,10,30,30,.9f,1};
    assert(decode(end, overlap, 2, 6, boxes) == 2 && boxes[0].class_id == 1 && boxes[1].x2 == 30); // no second NMS
    auto raw = dense; raw.output.shape = {1,6,2};
    assert(decode(raw, {20,20,20,20,20,20,20,20,.8f,.1f,.1f,.9f}, 6,2,boxes) == 1 && boxes[0].class_id == 1); // class-agnostic legacy NMS
    assert(decode(end, overlap, 2, 6, boxes, 1.f) == 0 && boxes.empty());
    auto malformed = overlap; malformed[5] = 2; assert(decode(end, malformed,2,6,boxes)<0 && boxes.empty());
    malformed = overlap; malformed[5] = .5f; assert(decode(end,malformed,2,6,boxes)<0);
    malformed = overlap; malformed[4] = 561; assert(decode(end,malformed,2,6,boxes)<0);
    malformed = overlap; malformed[4] = std::numeric_limits<float>::quiet_NaN(); assert(decode(end,malformed,2,6,boxes)<0);
    malformed = overlap; malformed[0] = std::numeric_limits<float>::infinity(); assert(decode(end,malformed,2,6,boxes)<0);
    malformed = overlap; malformed.pop_back(); assert(decode(end,malformed,2,6,boxes)<0);
    end.output.shape = {1,-1,6}; assert(decode(end,{},0,6,boxes)==0);
    assert(!validate_yolo_output(end,1,INT64_MAX,6,0));

    const auto graph_bytes = bytes(11,value_info("images",{1,3,320,320})) + bytes(12,value_info("output",{1,10,2100}));
    auto serialized = integer(1,8) + bytes(7,graph_bytes);
    YoloModelDescriptor parsed; std::string error;
    assert(inspect_yolo_model(serialized.data(),serialized.size(),&parsed,&error) && parsed.classes==6);
    for (size_t i=0;i<serialized.size();++i) assert(!inspect_yolo_model(serialized.data(),i,&parsed,&error));
    auto corrupt = serialized + std::string(11,static_cast<char>(0xff)); assert(!inspect_yolo_model(corrupt.data(),corrupt.size(),&parsed,&error));
    const auto entry = bytes(14,bytes(1,"task")+bytes(2,"detect"));
    corrupt = serialized + entry + entry; assert(!inspect_yolo_model(corrupt.data(),corrupt.size(),&parsed,&error));
    const auto nested = bytes(1,bytes(4,"NonMaxSuppression"));
    corrupt = bytes(7,graph_bytes + bytes(1,bytes(4,"If")+bytes(5,bytes(6,nested))));
    assert(!inspect_yolo_model(corrupt.data(),corrupt.size(),&parsed,&error));
    corrupt = serialized + bytes(25,bytes(7,bytes(4,"EfficientNMS_TRT")));
    assert(!inspect_yolo_model(corrupt.data(),corrupt.size(),&parsed,&error));
    std::cout << "YOLO descriptors, wire bounds, score semantics, layouts and NMS contracts passed\n";
}
