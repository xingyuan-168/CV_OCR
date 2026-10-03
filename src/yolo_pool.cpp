#include "yolo_pool.h"
#include "sha256.h"
#include "error.h"
#include <dxgi1_2.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <iomanip>

namespace ai {
namespace {
std::string environment(const char* key) {
    // DLLs use a private static CRT. getenv would see its startup snapshot and
    // miss settings changed by the host before loading a model.
    const std::wstring name(key,key+std::strlen(key));
    const DWORD needed=GetEnvironmentVariableW(name.c_str(),nullptr,0);
    if(!needed)return {};
    std::wstring value(needed,0);const DWORD size=GetEnvironmentVariableW(name.c_str(),value.data(),needed);
    if(!size ||size>=needed)return {};
    const int bytes=WideCharToMultiByte(CP_UTF8,0,value.data(),size,nullptr,0,nullptr,nullptr);
    std::string utf8(bytes,0);WideCharToMultiByte(CP_UTF8,0,value.data(),size,utf8.data(),bytes,nullptr,nullptr);return utf8;
}
int env_int(const char* key,int fallback) {
    const auto value=environment(key);if(value.empty())return fallback;
    size_t used=0;const int parsed=std::stoi(value,&used);if(used!=value.size())throw std::runtime_error(std::string("Invalid ")+key);return parsed;
}
std::string hardware_key() {
    std::string value=environment("PROCESSOR_IDENTIFIER")+"/cores="+std::to_string(physical_cores())+"/logical="+std::to_string(std::thread::hardware_concurrency());
    IDXGIFactory1* factory=nullptr;
    if(SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),reinterpret_cast<void**>(&factory)))) {
        IDXGIAdapter1* adapter=nullptr;
        for(UINT index=0;SUCCEEDED(factory->EnumAdapters1(index,&adapter));++index) {
            DXGI_ADAPTER_DESC1 description{};adapter->GetDesc1(&description);
            LARGE_INTEGER driver{};adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice),&driver);
            char name[1024]{};WideCharToMultiByte(CP_UTF8,0,description.Description,-1,name,sizeof(name),nullptr,nullptr);
            value+="/"+std::string(name)+"/"+std::to_string(description.VendorId)+"/"+std::to_string(description.DeviceId)+"/"+std::to_string(driver.QuadPart);
            adapter->Release(); adapter=nullptr;
        }
        factory->Release();
    }
    // Include all display devices and their registry driver identity; CUDA
    // ordinal need not be the same as the DXGI adapter ordinal on hybrid PCs.
    for(DWORD i=0;;++i) { DISPLAY_DEVICEW device{};device.cb=sizeof(device);if(!EnumDisplayDevicesW(nullptr,i,&device,0))break;
        char id[2048]{};WideCharToMultiByte(CP_UTF8,0,device.DeviceID,-1,id,sizeof(id),nullptr,nullptr);value+="/"+std::string(id);
    }
    return value;
}
struct Record {
    std::string key; int device=2,threads=1,fp16=0,graph=0,verified=0;
    double p50=0,p95=0;
};
std::string runtime_fingerprint() {
    static const std::string fingerprint=[] {
        std::string result;
        wchar_t filename[32768]{};
        HMODULE ort=GetModuleHandleW(L"onnxruntime.dll");
        if(ort &&GetModuleFileNameW(ort,filename,32768)) {
            auto directory=std::filesystem::path(filename).parent_path();
            for(const auto* name:{L"onnxruntime.dll",L"DirectML.dll"}) {
                std::ifstream file(directory/name,std::ios::binary);
                std::vector<char> bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
                result+=bytes.empty()?"missing":sha256(bytes.data(),bytes.size());
            }
        }
        HMODULE own=nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&yolo_device_name),&own);
        GetModuleFileNameW(own,filename,32768);
        std::ifstream file(std::filesystem::path(filename).parent_path()/L"CQ_YOLO_TensorRT.dll",std::ios::binary);
        std::vector<char> bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
        result+=bytes.empty()?"no-tensorrt-module":sha256(bytes.data(),bytes.size());
        return result;
    }();return fingerprint;
}
std::string calibration_key(const std::vector<uint8_t>& model,const YoloParameters& p) {
    const auto value="yolo-business-v4/ort=1.24.4/trt=10.13.3.9/cuda=12.8.90/conf=0.5/nms=0.45/model="+
        sha256(model.data(),model.size())+"/"+hardware_key()+"/input="+std::to_string(p.input_size)+
        "/slots="+std::to_string(p.session_count)+"/callers=5/workload="+p.workload_id+"/runtime="+runtime_fingerprint()+"/ordinal="+std::to_string(p.device_id);
    return sha256(value.data(),value.size());
}
bool read_record(const std::string& filename,const std::string& key,Record* result,bool business) {
    if(filename.empty())return false;
    std::ifstream file(std::filesystem::u8path(filename));std::string line;
    while(std::getline(file,line)) {
        std::istringstream row(line);Record r;
        if(!(row>>r.key>>r.device>>r.threads>>r.fp16>>r.graph>>r.verified>>r.p50>>r.p95) ||r.key!=key)continue;
        if(r.device<1 ||r.device>3 ||r.threads<1 ||r.threads>8 ||r.fp16<0 ||r.fp16>1 ||r.graph<0 ||r.graph>1 ||
            !std::isfinite(r.p50) ||!std::isfinite(r.p95) ||r.p50<=0 ||r.p95<r.p50 ||(business &&r.verified!=1))continue;
        if((r.fp16 ||r.graph) &&r.device!=AI_DEVICE_TENSORRT)continue;
        *result=r;return true;
    }
    return false;
}
std::filesystem::path short_cache() {
    wchar_t local[32768]{};GetEnvironmentVariableW(L"LOCALAPPDATA",local,32768);
    return std::filesystem::path(local)/L"CQ_AI"/L"yolo-short-v4.tsv";
}
void write_short(const Record& record) {
    static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
    try {
        auto path=short_cache();std::filesystem::create_directories(path.parent_path());
        std::ifstream input(path);std::vector<std::string> lines;std::string line;
        while(std::getline(input,line))if(line.substr(0,record.key.size())!=record.key)lines.push_back(line);
        std::ostringstream row;row<<record.key<<'\t'<<record.device<<'\t'<<record.threads<<"\t0\t0\t0\t"<<std::setprecision(12)<<record.p50<<'\t'<<record.p95;
        lines.push_back(row.str());auto temp=path;temp+=L"."+std::to_wstring(GetCurrentProcessId())+L".tmp";
        {std::ofstream output(temp);for(const auto& l:lines)output<<l<<'\n';output.flush();if(!output)return;}
        MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    }catch(...) {} // Optional performance cache must not prevent model loading.
}
std::shared_ptr<YoloPool> create_pool(const std::shared_ptr<const std::vector<uint8_t>>& model,Config config,
    const YoloParameters& p,int device,int threads,int fp16,int graph,int32_t* status,std::string* error) {
    auto pool=std::make_shared<YoloPool>();pool->model_bytes=model;pool->device_id=p.device_id;
    pool->session_count=p.session_count;pool->intra_op_threads=device==AI_DEVICE_CPU?threads:0;
    config.set_int("runtime.device_id",p.device_id);config.set_int("runtime.session_count",p.session_count);
    config.set_int("runtime.intra_op_threads",device==AI_DEVICE_CPU?threads:1);
    config.set_int("yolo.input_width",p.input_size);config.set_int("yolo.input_height",p.input_size);
    config.set_string("ocr.backend","null");
    config.set_string("yolo.precision",fp16?"fp16":"fp32");config.set_int("yolo.cuda_graph",graph);
    // Only this selector can authorize production precision/graph settings;
    // generic Engine config loads do not carry a qualified business record.
    config.set_int("yolo.validation_authorized",(fp16 ||graph)?1:0);
    config.set_string("yolo.engine_cache",p.engine_cache);
    const bool real=config.get_string("yolo.backend","onnxruntime")=="onnxruntime";
    if(real &&device==AI_DEVICE_TENSORRT)config.set_string("yolo.backend","tensorrt");
    pool->slots.reserve(p.session_count);
    for(int i=0;i<p.session_count;++i) {
        auto engine=std::make_unique<Engine>();
        if(!engine->init_ex(nullptr,device==AI_DEVICE_TENSORRT?AI_DEVICE_CPU:device,error)) {*status=AI_ERR_CONFIG;return nullptr;}
        *status=engine->yolo_load_model_from_memory_with_config(model->data(),static_cast<int32_t>(model->size()),config,device,error);
        if(*status<0)return nullptr;
        const auto runtime=get_thread_runtime_status();
        if(real &&runtime.active!=yolo_device_name(device)) {*status=AI_ERR_RUNTIME;*error="YOLO pool resolved to inconsistent backends";return nullptr;}
        if(i==0) {pool->runtime=runtime;pool->input_width=engine->yolo_input_width();pool->input_height=engine->yolo_input_height();}
        else if(pool->input_width!=engine->yolo_input_width() ||pool->input_height!=engine->yolo_input_height()) {*status=AI_ERR_RUNTIME;*error="YOLO pool input shape mismatch";return nullptr;}
        if(i>0 &&device==AI_DEVICE_TENSORRT)pool->runtime.cuda_graph=pool->runtime.cuda_graph &&runtime.cuda_graph;
        pool->slots.push_back({std::move(engine),false});
    }
    if(!real)pool->runtime.active=config.get_string("yolo.backend","mock");
    pool->runtime.execution_slots=p.session_count;*status=AI_OK;return pool;
}
Record short_benchmark(const std::shared_ptr<YoloPool>& pool,int device,int threads,const std::string& key) {
    Record record;record.key=key;record.device=device;record.threads=threads;
    std::vector<uint8_t> pixels(800*600*3,114);AIImage image{pixels.data(),800,600,2400,AI_IMAGE_BGR24};
    std::vector<double> samples[5];std::atomic<bool> go{false};std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for(int i=0;i<5;++i)workers.emplace_back([&,i] {
        while(!go.load(std::memory_order_acquire))std::this_thread::yield();
        std::vector<AIDetectBox> boxes;
        try {for(int n=0;n<11;++n) {
            const auto start=YoloClock::now();const int status=pool->detect(image,0.5f,&boxes);
            if(status<0){failed=true;return;}if(n>=2)samples[i].push_back(elapsed_us(start)/1000.0);
        }}catch(...){failed=true;}
    });
    go.store(true,std::memory_order_release);for(auto& worker:workers)worker.join();
    if(failed)throw std::runtime_error("YOLO synthetic calibration inference failed");
    std::vector<double> all;for(const auto& sample:samples)all.insert(all.end(),sample.begin(),sample.end());
    std::sort(all.begin(),all.end());record.p50=all[all.size()/2];record.p95=all[static_cast<size_t>((all.size()-1)*0.95)];return record;
}
bool better(const Record& candidate,const Record& current) {
    const bool meets=candidate.p95<=30,previous=current.p95<=30;
    if(meets!=previous)return meets;
    if(!meets &&std::abs(candidate.p95-current.p95)>current.p95*.05)return candidate.p95<current.p95;
    if(std::abs(candidate.p50-current.p50)>current.p50*.05)return candidate.p50<current.p50;
    // Within 5%, prefer fewer CPU threads or CPU over an extra GPU dependency.
    if(candidate.device==current.device)return candidate.threads<current.threads;
    return candidate.device==AI_DEVICE_CPU;
}
}
const char* yolo_device_name(int device) {
    switch(device){case 0:return "auto";case 1:return "directml";case 2:return "cpu";case 3:return "tensorrt";default:return "invalid";}
}
int physical_cores() {
    static const int count=[] {
        DWORD length=0;GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&length);
        std::vector<uint8_t> data(length);
        if(!length ||!GetLogicalProcessorInformationEx(RelationProcessorCore,reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()),&length))return static_cast<int>(std::max(1u,std::thread::hardware_concurrency()));
        int cores=0;for(size_t offset=0;offset<length;) {
            auto* entry=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()+offset);
            if(!entry->Size ||entry->Size>length-offset)break;++cores;offset+=entry->Size;
        }return std::max(1,cores);
    }();return count;
}
bool yolo_parameters(int input,int device,int ordinal,int sessions,YoloParameters* p,std::string* error,bool read_environment) {
    const char* invalid=nullptr;
    if(!p)invalid="YOLO parameters destination is null";
    else if(input!=0 &&input!=320 &&input!=640)invalid="input_size must be 0, 320, or 640";
    else if(device<0 ||device>3)invalid="runtime_device must be 0=AUTO, 1=DirectML, 2=CPU, or 3=TensorRT";
    else if(ordinal<0)invalid="device_id must be non-negative";
    else if(sessions<=0)invalid="session_count must be positive";
    if(invalid){if(error)*error=invalid;return false;}
    try {
        p->input_size=input;p->runtime_device=device;p->device_id=ordinal;p->session_count=sessions;
        if(!read_environment)return true;
        p->intra_op_threads=env_int("CQ_AI_YOLO_CPU_THREADS",0);
        if(p->intra_op_threads<0 ||p->intra_op_threads>8 ||
            ((device==AI_DEVICE_CPU ||device==AI_DEVICE_AUTO) &&p->intra_op_threads>1 &&p->intra_op_threads>physical_cores()/sessions))
            throw std::runtime_error("YOLO CPU threads exceed the physical-core budget");
        const auto precision=environment("CQ_AI_YOLO_PRECISION");
        if(!precision.empty() &&precision!="fp32" &&precision!="fp16")throw std::runtime_error("YOLO precision must be fp32 or fp16");
        p->fp16=precision=="fp16";p->graph=env_int("CQ_AI_YOLO_CUDA_GRAPH",0);p->validating=env_int("CQ_AI_YOLO_VALIDATING",0);
        if(p->graph<0 ||p->graph>1 ||p->validating<0 ||p->validating>1)throw std::runtime_error("YOLO graph/validation flags must be 0 or 1");
        p->calibration_file=environment("CQ_AI_YOLO_CALIBRATION_FILE");p->workload_id=environment("CQ_AI_YOLO_WORKLOAD_ID");p->engine_cache=environment("CQ_AI_YOLO_ENGINE_CACHE");
        return true;
    }catch(const std::exception& e){if(error)*error=e.what();return false;}
}
int YoloPool::acquire(Engine** engine) {
    const auto start=YoloClock::now();std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock,[&]{return std::any_of(slots.begin(),slots.end(),[](const YoloSlot& s){return !s.busy;});});
    for(size_t i=0;i<slots.size();++i)if(!slots[i].busy){slots[i].busy=true;*engine=slots[i].engine.get();yolo_timing.wait_us=elapsed_us(start);return static_cast<int>(i);}
    return AI_ERR_RUNTIME;
}
void YoloPool::release(int index) {{std::lock_guard<std::mutex> lock(mutex);slots[index].busy=false;}cv.notify_one();}
int32_t YoloPool::detect(const AIImage& image,float conf,std::vector<AIDetectBox>* output) {
    const auto start=YoloClock::now();Engine* engine=nullptr;const int index=acquire(&engine);if(index<0)return index;
    YoloLease lease(this,index);
    try {const auto status=engine->yolo_detect(image,conf,nms_threshold,output);last_latency_us=elapsed_us(start);return status;}
    catch(const std::exception& e){set_last_error(e.what());last_latency_us=elapsed_us(start);return AI_ERR_RUNTIME;}
}
std::shared_ptr<YoloPool> build_yolo_pool_shared(const std::shared_ptr<const std::vector<uint8_t>>& model,
    Config config,const YoloParameters& p,int32_t* status,std::string* error) {
    if(!model ||model->empty()){*status=AI_ERR_INVALID_ARGUMENT;*error="Empty YOLO model";return nullptr;}
    try {
        const auto key=calibration_key(*model,p);Record business;
        const bool qualified=read_record(p.calibration_file,key,&business,true) &&!p.workload_id.empty();
        const int budget=std::max(1,physical_cores()/p.session_count);
        if(config.get_string("yolo.backend","onnxruntime")!="onnxruntime") {
            auto pool=create_pool(model,config,p,p.runtime_device,p.intra_op_threads?p.intra_op_threads:1,0,0,status,error);
            if(pool){pool->runtime.calibration_key=key;pool->runtime.selection_basis="configured_mock";}return pool;
        }
        if(p.runtime_device!=AI_DEVICE_AUTO) {
            if((p.fp16 ||p.graph) &&p.runtime_device!=AI_DEVICE_TENSORRT) {
                *status=AI_ERR_CONFIG;*error="FP16 and CUDA Graph are only supported by the TensorRT YOLO backend";return nullptr;
            }
            const bool precision_qualified=qualified &&business.device==p.runtime_device &&business.fp16==p.fp16 &&business.graph==p.graph;
            if((p.fp16 ||p.graph) &&(!p.validating &&!precision_qualified)) {
                *status=AI_ERR_CONFIG;*error="FP16/CUDA Graph needs a matching validated business calibration record; use the offline validation tool first";return nullptr;
            }
            const int threads=p.intra_op_threads?p.intra_op_threads:(qualified &&business.device==p.runtime_device?business.threads:std::min(4,budget));
            auto pool=create_pool(model,config,p,p.runtime_device,threads,p.fp16,p.graph,status,error);
            if(pool){pool->runtime.requested=yolo_device_name(p.runtime_device);pool->runtime.calibration_key=key;pool->runtime.selection_basis=p.validating?"offline_validation":precision_qualified?"business_calibration":"explicit";set_runtime_status(pool->runtime);}return pool;
        }
        if(qualified &&business.threads<=budget) {
            auto pool=create_pool(model,config,p,business.device,business.threads,business.fp16,business.graph,status,error);
            if(pool){pool->runtime.requested="auto";pool->runtime.degraded=pool->runtime.active=="cpu";pool->runtime.calibration_key=key;pool->runtime.selection_basis="business_calibration";set_runtime_status(pool->runtime);return pool;}
        }
        Record cached;
        if(read_record(short_cache().u8string(),key,&cached,false) &&cached.threads<=budget) {
            auto pool=create_pool(model,config,p,cached.device,cached.threads,0,0,status,error);
            if(pool){pool->runtime.requested="auto";pool->runtime.degraded=pool->runtime.active=="cpu";pool->runtime.calibration_key=key;pool->runtime.selection_basis="synthetic_five_caller_cache";pool->runtime.reason="Provisional CPU/DirectML/TensorRT comparison; run real BMP business calibration";set_runtime_status(pool->runtime);return pool;}
        }
        std::shared_ptr<YoloPool> best;Record winner;std::string failures;
        for(int device:{AI_DEVICE_CPU,AI_DEVICE_DIRECTML,AI_DEVICE_TENSORRT}) {
            for(int threads:{1,2,4,6,8}) {
                if(threads>budget ||(device!=AI_DEVICE_CPU &&threads!=1) ||(p.intra_op_threads &&device==AI_DEVICE_CPU &&threads!=p.intra_op_threads))continue;
                std::string detail;auto pool=create_pool(model,config,p,device,threads,0,0,status,&detail);
                if(!pool){failures+=std::string(yolo_device_name(device))+": "+detail+"; ";continue;}
                const auto record=short_benchmark(pool,device,threads,key);
                if(!best ||better(record,winner)){best=pool;winner=record;}
            }
        }
        if(!best){*status=AI_ERR_RUNTIME;*error=failures;return nullptr;}
        best->runtime.requested="auto";best->runtime.degraded=winner.device==AI_DEVICE_CPU;best->runtime.calibration_key=key;best->runtime.selection_basis="synthetic_five_caller_benchmark";
        best->runtime.reason="Provisional CPU/DirectML/TensorRT comparison; run real BMP business calibration. "+failures;
        if(winner.device==AI_DEVICE_CPU)best->runtime.cpu_calibration_ms=winner.p50;
        if(winner.device==AI_DEVICE_DIRECTML)best->runtime.directml_calibration_ms=winner.p50;
        write_short(winner);*status=AI_OK;set_runtime_status(best->runtime);return best;
    }catch(const std::exception& e){*status=AI_ERR_RUNTIME;*error=e.what();return nullptr;}
}
} // namespace ai
