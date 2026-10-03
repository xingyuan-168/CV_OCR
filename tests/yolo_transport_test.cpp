#define NOMINMAX
#include "ai_engine.h"
#include "worker_protocol.h"
#include "pipe_io.h"
#include <cassert>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    assert(AI_InitEx(nullptr,AI_DEVICE_TENSORRT)<0);
    const auto read=[](const char* filename){std::ifstream file(filename,std::ios::binary);return std::vector<uint8_t>{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};};
    const auto model=read("models/yolo/best.onnx"), image=read("tests/fixtures/yolo/1.bmp");
    if(model.empty() ||image.empty()){std::cerr<<"Missing YOLO fixture";return 2;}
    DWORD topology_size=0;GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&topology_size);
    std::vector<uint8_t> topology(topology_size);int cores=0;
    if(GetLogicalProcessorInformationEx(RelationProcessorCore,reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(topology.data()),&topology_size))
        for(size_t offset=0;offset<topology_size;){const auto* entry=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(topology.data()+offset);++cores;offset+=entry->Size;}
    const auto initial_threads=std::to_wstring(std::max(1,std::min(8,cores)));
    SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",initial_threads.c_str());
    int32_t handle=0;assert(YOLO_Create(&handle)==AI_OK);
    assert(YOLO_LoadModelFromMemory(handle,model.data(),static_cast<int32_t>(model.size()),nullptr,0,0,AI_DEVICE_CPU,0,1)==AI_OK);
    assert(*YOLO_InferJson(handle,image.data(),static_cast<int32_t>(image.size()),.5f,0,0)=='[');
    // A second model must use its caller's settings even when the persistent
    // Worker was launched with an incompatible thread budget for two slots.
    SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",L"1");
    int32_t second=0;assert(YOLO_Create(&second)==AI_OK);
    assert(YOLO_LoadModelFromMemory(second,model.data(),static_cast<int32_t>(model.size()),nullptr,0,0,AI_DEVICE_CPU,0,2)==AI_OK);
    char second_status[8192]{};assert(YOLO_GetRuntimeStatusJson(second,second_status,sizeof(second_status))==AI_OK);
    assert(std::strstr(second_status,"\"intra_op_threads\":1"));assert(YOLO_Release(second)==AI_OK);
    SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",nullptr);
#if defined(_M_IX86)
    const auto connect=[](){HANDLE pipe=CreateFileA(ai_worker::pipe_name(ai_worker::RuntimeFlavor::Core),GENERIC_READ | GENERIC_WRITE,0,nullptr,OPEN_EXISTING,0,nullptr);assert(pipe!=INVALID_HANDLE_VALUE);return pipe;};
    // Reuse one connection for complete messages and validate correlation IDs.
    HANDLE idle=connect();
    for(uint64_t id:{1001ull,1002ull}) {
        ai_worker::Header request{ai_worker::kMagic,ai_worker::kVersion,ai_worker::CMD_RUNTIME_STATUS,0,id};
        assert(ai_worker::write_frame(idle,&request,sizeof(request)));
        ai_worker::ResponseHeader reply{};assert(ai_worker::read_frame(idle,&reply,sizeof(reply)));
        assert(reply.request_id==id &&reply.version==26 &&reply.status==AI_OK);
        std::vector<uint8_t> body(reply.payload_size);assert(ai_worker::read_frame(idle,body.data(),reply.payload_size));
    }
    HANDLE malformed=connect();
    ai_worker::Header bad{ai_worker::kMagic,ai_worker::kVersion,ai_worker::CMD_YOLO_INFER_JSON,ai_worker::kMaxPayload+1,1003};
    assert(ai_worker::write_frame(malformed,&bad,sizeof(bad)));
    ai_worker::ResponseHeader rejected{};assert(ai_worker::read_frame(malformed,&rejected,sizeof(rejected)));
    assert(rejected.status==AI_ERR_INVALID_ARGUMENT &&rejected.request_id==1003);CloseHandle(malformed);
    HANDLE partial=connect();bad.payload_size=100;bad.request_id=1004;
    assert(ai_worker::write_frame(partial,&bad,sizeof(bad)));const uint8_t byte=1;
    assert(ai_worker::write_frame(partial,&byte,1));CloseHandle(partial);
    assert(*YOLO_InferJson(handle,image.data(),static_cast<int32_t>(image.size()),.5f,0,0)=='[');
    // Idle long connections must not make graceful shutdown wait forever.
    assert(AI_ShutdownWorker()==AI_OK);CloseHandle(idle);
    assert(*YOLO_InferJson(handle,image.data(),static_cast<int32_t>(image.size()),.5f,0,0)==0);
    assert(AI_GetLastError()[0]);
    assert(YOLO_Create(&handle)==AI_OK);
    assert(YOLO_LoadModelFromMemory(handle,model.data(),static_cast<int32_t>(model.size()),nullptr,0,0,AI_DEVICE_CPU,0,1)==AI_OK);
#endif
    // Concurrent release can invalidate later requests; requests holding a pool
    // must finish safely and never borrow a destroyed Session.
    std::vector<std::thread> callers;
    for(int i=0;i<5;++i)callers.emplace_back([&]{for(int n=0;n<10;++n){const char* json=YOLO_InferJson(handle,image.data(),static_cast<int32_t>(image.size()),.5f,0,0);assert(json &&(*json=='[' ||*json==0));}});
    Sleep(5);assert(YOLO_Release(handle)==AI_OK);for(auto& caller:callers)caller.join();
    assert(*YOLO_InferJson(handle,image.data(),static_cast<int32_t>(image.size()),.5f,0,0)==0);
    // Missing NVIDIA driver must leave a failed handle available for CPU retry.
    HMODULE driver=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!driver){
        SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",L"8");
        int32_t optional=0;assert(YOLO_Create(&optional)==AI_OK);
        assert(YOLO_LoadModelFromMemory(optional,model.data(),static_cast<int32_t>(model.size()),nullptr,0,0,AI_DEVICE_TENSORRT,0,20)==AI_ERR_RUNTIME);
        assert(AI_GetLastError()[0]);
        SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",nullptr);
        assert(YOLO_LoadModelFromMemory(optional,model.data(),static_cast<int32_t>(model.size()),nullptr,0,0,AI_DEVICE_CPU,0,1)==AI_OK);
        assert(YOLO_Release(optional)==AI_OK);
    }else FreeLibrary(driver);
    assert(AI_ShutdownWorker()==AI_OK);
    std::cout<<"YOLO transport, lifecycle, request correlation and optional-dependency regression passed\n";
}
