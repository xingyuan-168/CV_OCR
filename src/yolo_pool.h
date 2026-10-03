#pragma once
#include "engine.h"
#include "runtime_status.h"
#include "yolo_diagnostics.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <atomic>
namespace ai {
struct YoloParameters {
    int32_t input_size=0, runtime_device=AI_DEVICE_CPU, device_id=0, session_count=1, intra_op_threads=0;
    int32_t fp16=0, graph=0, validating=0;
    std::string calibration_file, workload_id, engine_cache;
};
const char* yolo_device_name(int32_t device);
int physical_cores();
bool yolo_parameters(int input, int device, int ordinal, int sessions, YoloParameters*, std::string*, bool read_environment=true);
struct YoloSlot { std::unique_ptr<Engine> engine; bool busy=false; };
struct YoloPool {
    std::mutex mutex; std::condition_variable cv;
    std::vector<YoloSlot> slots;
    std::shared_ptr<const std::vector<uint8_t>> model_bytes;
    RuntimeStatus runtime;
    int32_t input_width=0,input_height=0,device_id=0,session_count=0,intra_op_threads=0;
    float nms_threshold=0.45f;
    std::atomic<int64_t> last_latency_us{-1};
    int acquire(Engine** engine);
    void release(int index);
    int32_t detect(const AIImage&,float,std::vector<AIDetectBox>*);
};
struct YoloLease {
    YoloPool* pool; int index;
    ~YoloLease() { if(pool && index>=0) pool->release(index); }
    YoloLease(const YoloLease&)=delete;
    YoloLease(YoloPool* p,int i):pool(p),index(i){}
};
std::shared_ptr<YoloPool> build_yolo_pool_shared(const std::shared_ptr<const std::vector<uint8_t>>&,
    Config, const YoloParameters&, int32_t*, std::string*);
} // namespace ai
