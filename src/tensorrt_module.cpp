#include "tensorrt_module_api.h"
#include "sha256.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using namespace nvinfer1;
using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& error) { if (!condition) throw std::runtime_error(error); }
template<class T> T symbol(HMODULE module, const char* name) {
    const auto result = reinterpret_cast<T>(GetProcAddress(module, name));
    require(result != nullptr, std::string("NVIDIA entry point missing: ") + name);
    return result;
}
struct Logger : ILogger {
    std::mutex mutex; std::string error;
    void log(Severity severity, const char* message) noexcept override {
        if (severity > Severity::kWARNING) return;
        try { std::lock_guard<std::mutex> lock(mutex); error = message ? message : "TensorRT error"; } catch (...) {}
    }
};
struct Libraries {
    HMODULE driver = nullptr, infer = nullptr, parser = nullptr, runtime = nullptr, plugins = nullptr;
    DLL_DIRECTORY_COOKIE directory_cookie = nullptr;
    Logger logger;
#define CUDA_FUNCTION(name) decltype(&name) name##_fn = nullptr
    CUDA_FUNCTION(cuInit); CUDA_FUNCTION(cuDeviceGet); CUDA_FUNCTION(cuDeviceGetCount);
    CUDA_FUNCTION(cuDeviceGetName); CUDA_FUNCTION(cuDeviceGetUuid); CUDA_FUNCTION(cuDeviceGetAttribute);
    CUDA_FUNCTION(cuDriverGetVersion); CUDA_FUNCTION(cuDevicePrimaryCtxRetain); CUDA_FUNCTION(cuDevicePrimaryCtxRelease);
    CUDA_FUNCTION(cuCtxPushCurrent); CUDA_FUNCTION(cuCtxPopCurrent);
    CUDA_FUNCTION(cuMemAlloc); CUDA_FUNCTION(cuMemFree); CUDA_FUNCTION(cuMemHostAlloc); CUDA_FUNCTION(cuMemFreeHost);
    CUDA_FUNCTION(cuStreamCreate); CUDA_FUNCTION(cuStreamDestroy); CUDA_FUNCTION(cuStreamSynchronize);
    CUDA_FUNCTION(cuMemcpyHtoDAsync); CUDA_FUNCTION(cuMemcpyDtoHAsync);
    CUDA_FUNCTION(cuEventCreate); CUDA_FUNCTION(cuEventDestroy); CUDA_FUNCTION(cuEventRecord); CUDA_FUNCTION(cuEventElapsedTime);
    CUDA_FUNCTION(cuStreamBeginCapture); CUDA_FUNCTION(cuStreamEndCapture);
    CUDA_FUNCTION(cuGraphInstantiateWithFlags); CUDA_FUNCTION(cuGraphLaunch); CUDA_FUNCTION(cuGraphDestroy); CUDA_FUNCTION(cuGraphExecDestroy);
#undef CUDA_FUNCTION
    decltype(&createInferBuilder_INTERNAL) builder_fn = nullptr;
    decltype(&createInferRuntime_INTERNAL) runtime_fn = nullptr;
    decltype(&createNvOnnxParser_INTERNAL) parser_fn = nullptr;
    int cuda_runtime_version = 0;
    std::string driver_file_version, driver_binary_sha256;
    Libraries() {
      try {
        // Retain modules for the lifetime of every engine/context. No NVIDIA
        // import library is linked into the base DLL, Worker, or this module.
        driver = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(driver != nullptr, "NVIDIA driver nvcuda.dll is missing");
        wchar_t driver_path[32768]{};
        require(GetModuleFileNameW(driver,driver_path,32768)!=0,"Cannot identify loaded NVIDIA driver");
        std::ifstream driver_file(std::filesystem::path(driver_path),std::ios::binary);
        const std::vector<char> driver_bytes{std::istreambuf_iterator<char>(driver_file),std::istreambuf_iterator<char>()};
        require(!driver_bytes.empty(),"Cannot hash loaded NVIDIA driver");
        driver_binary_sha256=ai::sha256(driver_bytes.data(),driver_bytes.size());
        DWORD ignored=0;const DWORD version_size=GetFileVersionInfoSizeW(driver_path,&ignored);
        std::vector<char> version_bytes(version_size);
        VS_FIXEDFILEINFO* fixed=nullptr;UINT fixed_size=0;
        if(version_size &&GetFileVersionInfoW(driver_path,0,version_size,version_bytes.data()) &&
            VerQueryValueW(version_bytes.data(),L"\\",reinterpret_cast<void**>(&fixed),&fixed_size) &&fixed_size>=sizeof(*fixed)) {
            driver_file_version=std::to_string(HIWORD(fixed->dwFileVersionMS))+"."+std::to_string(LOWORD(fixed->dwFileVersionMS))+"."+
                std::to_string(HIWORD(fixed->dwFileVersionLS))+"."+std::to_string(LOWORD(fixed->dwFileVersionLS));
        }
#define LOAD_CUDA(name, exported) name##_fn = symbol<decltype(name##_fn)>(driver, exported)
        LOAD_CUDA(cuInit,"cuInit"); LOAD_CUDA(cuDeviceGet,"cuDeviceGet"); LOAD_CUDA(cuDeviceGetCount,"cuDeviceGetCount");
        LOAD_CUDA(cuDeviceGetName,"cuDeviceGetName"); LOAD_CUDA(cuDeviceGetUuid,"cuDeviceGetUuid"); LOAD_CUDA(cuDeviceGetAttribute,"cuDeviceGetAttribute");
        LOAD_CUDA(cuDriverGetVersion,"cuDriverGetVersion"); LOAD_CUDA(cuDevicePrimaryCtxRetain,"cuDevicePrimaryCtxRetain");
        LOAD_CUDA(cuDevicePrimaryCtxRelease,"cuDevicePrimaryCtxRelease_v2"); LOAD_CUDA(cuCtxPushCurrent,"cuCtxPushCurrent_v2"); LOAD_CUDA(cuCtxPopCurrent,"cuCtxPopCurrent_v2");
        LOAD_CUDA(cuMemAlloc,"cuMemAlloc_v2"); LOAD_CUDA(cuMemFree,"cuMemFree_v2"); LOAD_CUDA(cuMemHostAlloc,"cuMemHostAlloc"); LOAD_CUDA(cuMemFreeHost,"cuMemFreeHost");
        LOAD_CUDA(cuStreamCreate,"cuStreamCreate"); LOAD_CUDA(cuStreamDestroy,"cuStreamDestroy_v2"); LOAD_CUDA(cuStreamSynchronize,"cuStreamSynchronize");
        LOAD_CUDA(cuMemcpyHtoDAsync,"cuMemcpyHtoDAsync_v2"); LOAD_CUDA(cuMemcpyDtoHAsync,"cuMemcpyDtoHAsync_v2");
        LOAD_CUDA(cuEventCreate,"cuEventCreate"); LOAD_CUDA(cuEventDestroy,"cuEventDestroy_v2"); LOAD_CUDA(cuEventRecord,"cuEventRecord"); LOAD_CUDA(cuEventElapsedTime,"cuEventElapsedTime");
        LOAD_CUDA(cuStreamBeginCapture,"cuStreamBeginCapture"); LOAD_CUDA(cuStreamEndCapture,"cuStreamEndCapture");
        LOAD_CUDA(cuGraphInstantiateWithFlags,"cuGraphInstantiateWithFlags"); LOAD_CUDA(cuGraphLaunch,"cuGraphLaunch");
        LOAD_CUDA(cuGraphDestroy,"cuGraphDestroy"); LOAD_CUDA(cuGraphExecDestroy,"cuGraphExecDestroy");
#undef LOAD_CUDA
        HMODULE own = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&symbol<void*>), &own);
        wchar_t filename[32768]{}; GetModuleFileNameW(own,filename,32768);
        auto directory = std::filesystem::path(filename).parent_path() / L"nvidia";
        wchar_t configured[32768]{};
        if (GetEnvironmentVariableW(L"CQ_AI_NVIDIA_DIR", configured, 32768)) directory = configured;
        if (!std::filesystem::is_directory(directory)) directory = std::filesystem::path(filename).parent_path();
        directory_cookie = AddDllDirectory(directory.c_str());
        const auto load = [&](const wchar_t* name) {
            const auto path = directory / name;
            const auto module = LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            require(module != nullptr, "Cannot load " + path.u8string() + " (Windows error " + std::to_string(GetLastError()) + ")"); return module;
        };
        infer = load(L"nvinfer_10.dll"); parser = load(L"nvonnxparser_10.dll"); runtime = load(L"cudart64_12.dll");
        const auto version = symbol<decltype(&getInferLibVersion)>(infer,"getInferLibVersion")();
        require(version == NV_TENSORRT_VERSION, "TensorRT runtime must match build headers 10.13.3; got " + std::to_string(version));
        const auto cuda_version = symbol<int (*)(int*)>(runtime,"cudaRuntimeGetVersion");
        require(cuda_version(&cuda_runtime_version) == 0 && cuda_runtime_version == 12080, "CUDA runtime must be 12.8 Update 1");
        builder_fn = symbol<decltype(builder_fn)>(infer,"createInferBuilder_INTERNAL");
        runtime_fn = symbol<decltype(runtime_fn)>(infer,"createInferRuntime_INTERNAL");
        parser_fn = symbol<decltype(parser_fn)>(parser,"createNvOnnxParser_INTERNAL");
        // Standard YOLO needs no custom plugin, but initialize the vendor
        // registry when its optional library is included in the dependency pack.
        plugins = LoadLibraryExW((directory/L"nvinfer_plugin_10.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (plugins) {
            using InitPlugins = bool (*)(void*, const char*);
            auto init = symbol<InitPlugins>(plugins,"initLibNvInferPlugins");
            require(init(&logger,""), "TensorRT plugin registry initialization failed");
        }
      } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (plugins) FreeLibrary(plugins); if (parser) FreeLibrary(parser);
        if (infer) FreeLibrary(infer); if (runtime) FreeLibrary(runtime); if (driver) FreeLibrary(driver);
        if (directory_cookie) RemoveDllDirectory(directory_cookie);
    }
    ~Libraries() { cleanup(); }
};
Libraries& libraries() { static Libraries value; return value; }
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) throw std::runtime_error(std::string(operation) + " failed (CUDA driver status " + std::to_string(result) + ")");
}
struct Current {
    explicit Current(CUcontext context) { check(libraries().cuCtxPushCurrent_fn(context),"cuCtxPushCurrent"); }
    ~Current() { CUcontext previous{}; libraries().cuCtxPopCurrent_fn(&previous); }
};
std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto length = file.tellg(); if (length <= 0 || length > 1024ll*1024*1024) return {};
    std::vector<char> data(static_cast<size_t>(length)); file.seekg(0); file.read(data.data(),length);
    return file ? data : std::vector<char>{};
}
void atomic_file(const std::filesystem::path& path, const void* data, size_t length) {
    auto temporary = path; temporary += L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
    { std::ofstream file(temporary,std::ios::binary | std::ios::trunc); file.write(static_cast<const char*>(data),length);
      file.flush(); require(static_cast<bool>(file),"TensorRT cache write failed"); }
    require(MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,"TensorRT cache atomic rename failed");
}
std::vector<char> verified_file(const std::filesystem::path& path) {
    auto data = read_file(path); if (data.empty()) return {};
    auto digest = read_file(path.wstring() + L".sha256");
    if (std::string(digest.begin(),digest.end()) != ai::sha256(data.data(),data.size())) return {};
    return data;
}
void save_verified(const std::filesystem::path& path, const void* data, size_t size) {
    const auto digest = ai::sha256(data,size);
    atomic_file(path,data,size); atomic_file(path.wstring()+L".sha256",digest.data(),digest.size());
}
struct Engine {
    CUdevice device{}; CUcontext primary{};
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    ai_trt::Info info;
    std::string input,output;
    ~Engine() {
        if (!primary) return;
        try { Current current(primary); engine.reset(); runtime.reset(); } catch (...) {}
        libraries().cuDevicePrimaryCtxRelease_fn(device);
    }
};
std::mutex engine_mutex;
std::unordered_map<std::string,std::weak_ptr<Engine>> engines;
std::shared_ptr<Engine> shared_engine(const void* model, size_t size, const ai_trt::Options& options) {
    auto& lib = libraries(); check(lib.cuInit_fn(0),"cuInit");
    auto shared = std::make_shared<Engine>();
    int count = 0; check(lib.cuDeviceGetCount_fn(&count),"cuDeviceGetCount");
    require(options.device >= 0 && options.device < count,"TensorRT CUDA device ordinal is unavailable");
    check(lib.cuDeviceGet_fn(&shared->device,options.device),"cuDeviceGet");
    check(lib.cuDeviceGetName_fn(shared->info.gpu,sizeof(shared->info.gpu),shared->device),"cuDeviceGetName");
    CUuuid uuid{}; check(lib.cuDeviceGetUuid_fn(&uuid,shared->device),"cuDeviceGetUuid");
    check(lib.cuDriverGetVersion_fn(&shared->info.driver_version),"cuDriverGetVersion");
    require(shared->info.driver_version >= 12080,"Update NVIDIA driver: this fixed baseline requires CUDA driver capability >=12.8");
    int major = 0, minor = 0;
    check(lib.cuDeviceGetAttribute_fn(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,shared->device),"GPU compute capability");
    check(lib.cuDeviceGetAttribute_fn(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,shared->device),"GPU compute capability");
    require(major*10+minor >= 75,"TensorRT 10.13.3 requires compute capability >=7.5");
    shared->info.cuda_version = lib.cuda_runtime_version; shared->info.fp16 = options.fp16;
    std::strcpy(shared->info.driver_file_version,lib.driver_file_version.c_str());
    std::strcpy(shared->info.driver_binary_sha256,lib.driver_binary_sha256.c_str());
    std::strcpy(shared->info.version,"10.13.3.9");
    const std::string metadata = "cq-yolo-engine-v3\nmodel=" + ai::sha256(model,size) + "\ngpu=" + shared->info.gpu +
        "\nuuid=" + ai::sha256(&uuid,sizeof(uuid)) + "\ncc=" + std::to_string(major*10+minor) +
        "\ndriver=" + std::to_string(shared->info.driver_version) + "\ndriver_file=" + lib.driver_file_version +
        "\ndriver_sha256=" + lib.driver_binary_sha256 + "\ncuda=12.8.90\ntrt=10.13.3.9\nfp16=" +
        std::to_string(options.fp16) + "\nworkspace=" + std::to_string(options.workspace) + "\nbatch=1\nio=fp32\ntf32=0\naux=0\n";
    const auto key = ai::sha256(metadata.data(),metadata.size()); std::strcpy(shared->info.cache_key,key.c_str());
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (auto existing = engines[key].lock()) return existing;
    check(lib.cuDevicePrimaryCtxRetain_fn(&shared->primary,shared->device),"cuDevicePrimaryCtxRetain");
    Current current(shared->primary);
    std::filesystem::path directory;
    if (options.cache_directory) directory = options.cache_directory;
    else { wchar_t local[32768]{}; GetEnvironmentVariableW(L"LOCALAPPDATA",local,32768); directory=std::filesystem::path(local)/L"CQ_AI"/L"tensorrt-v3"; }
    std::filesystem::create_directories(directory);
    const auto plan_path=directory/(key+".plan"), timing_path=directory/(key+".timing");
    // Serialize builders across processes sharing a cache; corrupt entries are
    // hash-checked before deserialize and replaced after a successful build.
    HANDLE cache_mutex=CreateMutexW(nullptr,FALSE,(L"Local\\CQ_YOLO_ENGINE_"+std::wstring(key.begin(),key.end())).c_str());
    struct CacheLock { HANDLE handle; ~CacheLock(){ if(handle){ReleaseMutex(handle);CloseHandle(handle);} } } cache_lock{cache_mutex};
    require(cache_mutex != nullptr,"Cannot create TensorRT cache mutex");
    const DWORD wait=WaitForSingleObject(cache_mutex,10*60*1000);
    require(wait==WAIT_OBJECT_0 || wait==WAIT_ABANDONED,"Timed out waiting for TensorRT engine builder");
    shared->runtime.reset(static_cast<IRuntime*>(lib.runtime_fn(&lib.logger,NV_TENSORRT_VERSION)));
    require(shared->runtime!=nullptr,"createInferRuntime failed");
    auto plan=verified_file(plan_path);
    if (!plan.empty()) shared->engine.reset(shared->runtime->deserializeCudaEngine(plan.data(),plan.size()));
    shared->info.cache_hit = shared->engine != nullptr;
    if (!shared->engine) {
        const auto start=Clock::now();
        std::unique_ptr<IBuilder> builder(static_cast<IBuilder*>(lib.builder_fn(&lib.logger,NV_TENSORRT_VERSION)));
        require(builder!=nullptr,"createInferBuilder failed");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
        std::unique_ptr<nvonnxparser::IParser> parser(static_cast<nvonnxparser::IParser*>(lib.parser_fn(network.get(),&lib.logger,NV_ONNX_PARSER_VERSION)));
        require(parser && parser->parse(model,size),"TensorRT ONNX parse failed: "+lib.logger.error);
        require(network->getNbInputs()==1 && network->getNbOutputs()==1,"TensorRT YOLO requires exactly one input and one output");
        require(network->getInput(0)->getType()==DataType::kFLOAT,"TensorRT YOLO input must be FP32");
        network->getInput(0)->setAllowedFormats(1u<<static_cast<uint32_t>(TensorFormat::kLINEAR));
        network->getOutput(0)->setType(DataType::kFLOAT);
        network->getOutput(0)->setAllowedFormats(1u<<static_cast<uint32_t>(TensorFormat::kLINEAR));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,options.workspace);
        config->clearFlag(BuilderFlag::kTF32); config->setMaxAuxStreams(0);
        if (options.fp16) { require(builder->platformHasFastFp16(),"GPU lacks fast FP16"); config->setFlag(BuilderFlag::kFP16); }
        auto timing=verified_file(timing_path);
        std::unique_ptr<ITimingCache> cache(config->createTimingCache(timing.data(),timing.size()));
        require(cache && config->setTimingCache(*cache,false),"Cannot attach TensorRT timing cache");
        std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network,*config));
        require(serialized!=nullptr,"TensorRT engine build failed: "+lib.logger.error);
        shared->engine.reset(shared->runtime->deserializeCudaEngine(serialized->data(),serialized->size()));
        require(shared->engine!=nullptr,"Cannot deserialize freshly built TensorRT engine");
        save_verified(plan_path,serialized->data(),serialized->size());
        std::unique_ptr<IHostMemory> timing_data(cache->serialize());
        if (timing_data) save_verified(timing_path,timing_data->data(),timing_data->size());
        atomic_file(directory/(key+".manifest"),metadata.data(),metadata.size());
        shared->info.build_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
    }
    require(shared->engine->getNbIOTensors()==2,"TensorRT YOLO engine I/O count mismatch");
    for (int i=0;i<2;++i) {
        const char* name=shared->engine->getIOTensorName(i);
        require(shared->engine->getTensorDataType(name)==DataType::kFLOAT,"TensorRT I/O must remain FP32");
        require(shared->engine->getTensorFormat(name)==TensorFormat::kLINEAR,"TensorRT I/O must be contiguous linear FP32");
        if (shared->engine->getTensorIOMode(name)==TensorIOMode::kINPUT) shared->input=name;
        else shared->output=name;
    }
    const auto in=shared->engine->getTensorShape(shared->input.c_str());
    const auto out=shared->engine->getTensorShape(shared->output.c_str());
    require(in.nbDims==4 && in.d[0]==1 && in.d[1]==3 && in.d[2]>0 && in.d[2]==in.d[3],"Only static square NCHW batch=1 YOLO is supported");
    require(out.nbDims==3 && out.d[0]==1 && out.d[1]>4 && out.d[2]>0,"YOLO output must be [1,4+classes,candidates]");
    require(in.d[2]<=4096 && out.d[1]<=65536 && out.d[2]<=1000000,"TensorRT YOLO tensor exceeds allocation limits");
    require(static_cast<uint64_t>(out.d[1])*out.d[2]*sizeof(float)<=512ull*1024*1024,"TensorRT YOLO output exceeds 512 MiB allocation limit");
    shared->info.height=static_cast<int32_t>(in.d[2]); shared->info.width=static_cast<int32_t>(in.d[3]);
    shared->info.attrs=static_cast<int32_t>(out.d[1]); shared->info.candidates=static_cast<int32_t>(out.d[2]);
    engines[key]=shared; return shared;
}
struct Slot {
    std::shared_ptr<Engine> shared;
    std::unique_ptr<IExecutionContext> context;
    CUstream stream{}; CUdeviceptr device_input{},device_output{};
    float* host_input=nullptr; float* host_output=nullptr;
    size_t input_bytes=0,output_bytes=0;
    CUevent events[4]{}; CUgraph graph{}; CUgraphExec graph_exec{};
    ~Slot() {
        if (!shared) return;
        try {
            Current current(shared->primary); auto& lib=libraries();
            if(stream) lib.cuStreamSynchronize_fn(stream);
            if(graph_exec) lib.cuGraphExecDestroy_fn(graph_exec); if(graph) lib.cuGraphDestroy_fn(graph);
            context.reset();
            for(auto event:events) if(event) lib.cuEventDestroy_fn(event);
            if(device_input) lib.cuMemFree_fn(device_input); if(device_output) lib.cuMemFree_fn(device_output);
            if(host_input) lib.cuMemFreeHost_fn(host_input); if(host_output) lib.cuMemFreeHost_fn(host_output);
            if(stream) lib.cuStreamDestroy_fn(stream);
        } catch (...) {}
    }
};
void error_text(char* output,size_t capacity,const char* error) {
    if(output && capacity) { std::strncpy(output,error,capacity-1); output[capacity-1]=0; }
}
void* create(const void* model,size_t size,const ai_trt::Options* options,ai_trt::Info* info,char* error,size_t capacity) {
    try {
        require(model && size && options && info && options->abi==ai_trt::kAbi,"TensorRT module create arguments/ABI invalid");
        auto slot=std::make_unique<Slot>(); slot->shared=shared_engine(model,size,*options);
        Current current(slot->shared->primary); auto& lib=libraries();
        slot->context.reset(slot->shared->engine->createExecutionContext());
        require(slot->context!=nullptr,"TensorRT execution context allocation failed");
        check(lib.cuStreamCreate_fn(&slot->stream,CU_STREAM_NON_BLOCKING),"cuStreamCreate");
        slot->input_bytes=static_cast<size_t>(3)*slot->shared->info.width*slot->shared->info.height*sizeof(float);
        slot->output_bytes=static_cast<size_t>(slot->shared->info.attrs)*slot->shared->info.candidates*sizeof(float);
        check(lib.cuMemHostAlloc_fn(reinterpret_cast<void**>(&slot->host_input),slot->input_bytes,0),"pinned input allocation");
        check(lib.cuMemHostAlloc_fn(reinterpret_cast<void**>(&slot->host_output),slot->output_bytes,0),"pinned output allocation");
        check(lib.cuMemAlloc_fn(&slot->device_input,slot->input_bytes),"device input allocation");
        check(lib.cuMemAlloc_fn(&slot->device_output,slot->output_bytes),"device output allocation");
        require(slot->context->setTensorAddress(slot->shared->input.c_str(),reinterpret_cast<void*>(slot->device_input)) &&
            slot->context->setTensorAddress(slot->shared->output.c_str(),reinterpret_cast<void*>(slot->device_output)),"TensorRT setTensorAddress failed");
        for(auto& event:slot->events) check(lib.cuEventCreate_fn(&event,CU_EVENT_DEFAULT),"cuEventCreate");
        *info=slot->shared->info;
        if(options->graph) {
            // Flush lazy setup before capture; capture failure is not fatal.
            std::memset(slot->host_input,0,slot->input_bytes);
            check(lib.cuMemcpyHtoDAsync_fn(slot->device_input,slot->host_input,slot->input_bytes,slot->stream),"graph warmup copy");
            require(slot->context->enqueueV3(reinterpret_cast<cudaStream_t>(slot->stream)),"graph warmup enqueue failed");
            check(lib.cuStreamSynchronize_fn(slot->stream),"graph warmup synchronize");
            if(lib.cuStreamBeginCapture_fn(slot->stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)==CUDA_SUCCESS) {
                const bool enqueued=slot->context->enqueueV3(reinterpret_cast<cudaStream_t>(slot->stream));
                const auto result=lib.cuStreamEndCapture_fn(slot->stream,&slot->graph);
                if(enqueued && result==CUDA_SUCCESS && slot->graph &&
                    lib.cuGraphInstantiateWithFlags_fn(&slot->graph_exec,slot->graph,0)==CUDA_SUCCESS) info->graph=1;
                else { if(slot->graph) lib.cuGraphDestroy_fn(slot->graph); slot->graph=nullptr; }
            }
        }
        return slot.release();
    } catch(const std::exception& e) { error_text(error,capacity,e.what()); return nullptr; }
    catch(...) { error_text(error,capacity,"Unexpected TensorRT create failure"); return nullptr; }
}
void destroy(void* handle) { delete static_cast<Slot*>(handle); }
float* input(void* handle) { return static_cast<Slot*>(handle)->host_input; }
const float* output(void* handle) { return static_cast<Slot*>(handle)->host_output; }
int32_t run(void* handle,ai_trt::Timing* timing,char* error,size_t capacity) {
    try {
        require(handle && timing,"Invalid TensorRT execution arguments");
        auto& slot=*static_cast<Slot*>(handle); Current current(slot.shared->primary); auto& lib=libraries();
        check(lib.cuEventRecord_fn(slot.events[0],slot.stream),"record H2D start");
        check(lib.cuMemcpyHtoDAsync_fn(slot.device_input,slot.host_input,slot.input_bytes,slot.stream),"H2D");
        check(lib.cuEventRecord_fn(slot.events[1],slot.stream),"record GPU start");
        if(slot.graph_exec) check(lib.cuGraphLaunch_fn(slot.graph_exec,slot.stream),"CUDA Graph launch");
        else require(slot.context->enqueueV3(reinterpret_cast<cudaStream_t>(slot.stream)),"TensorRT enqueueV3 failed");
        check(lib.cuEventRecord_fn(slot.events[2],slot.stream),"record D2H start");
        check(lib.cuMemcpyDtoHAsync_fn(slot.host_output,slot.device_output,slot.output_bytes,slot.stream),"D2H");
        check(lib.cuEventRecord_fn(slot.events[3],slot.stream),"record D2H end");
        check(lib.cuStreamSynchronize_fn(slot.stream),"request stream synchronize");
        int64_t* fields[]={&timing->h2d_us,&timing->gpu_us,&timing->d2h_us};
        for(int i=0;i<3;++i){ float ms=0;check(lib.cuEventElapsedTime_fn(&ms,slot.events[i],slot.events[i+1]),"CUDA event elapsed time"); *fields[i]=static_cast<int64_t>(ms*1000); }
        return 0;
    } catch(const std::exception& e) { error_text(error,capacity,e.what()); return -1; }
    catch(...) { error_text(error,capacity,"Unexpected TensorRT inference failure"); return -1; }
}
const ai_trt::Api api{ai_trt::kAbi,create,destroy,input,output,run};
}
extern "C" __declspec(dllexport) const ai_trt::Api* CQ_YOLO_TensorRT_GetApi(uint32_t abi) {
    return abi==ai_trt::kAbi ? &api : nullptr;
}
