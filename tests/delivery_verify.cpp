#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define AIENGINE_EXPORT extern "C"
#include "ai_engine.h"
#include "worker_protocol.h"
#include "pipe_io.h"
#include "sha256.h"
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

namespace {
void require(bool value, const std::string& error) { if (!value) throw std::runtime_error(error); }
std::vector<uint8_t> bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot read file: " + path.u8string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
std::string narrow(const std::wstring& value, UINT codepage=CP_UTF8) {
    const int n=WideCharToMultiByte(codepage,0,value.data(),static_cast<int>(value.size()),nullptr,0,nullptr,nullptr);
    std::string result(n,0); WideCharToMultiByte(codepage,0,value.data(),static_cast<int>(value.size()),result.data(),n,nullptr,nullptr); return result;
}
std::string quoted(const std::string& value) {
    std::string result="\"";
    for(unsigned char c:value) { if(c=='\\'||c=='\"')result+='\\'; if(c=='\n'){result+="\\n";continue;} if(c=='\r'){result+="\\r";continue;} result+=c; }
    return result+'\"';
}
template<class T> T function(HMODULE module,const char* name) {
    const auto value=GetProcAddress(module,name); require(value!=nullptr,std::string("Missing export: ")+name); return reinterpret_cast<T>(value);
}
struct Library { HMODULE value=nullptr; ~Library(){if(value)FreeLibrary(value);} };
struct ModelGuard {
    int32_t handle=0;
    decltype(&YOLO_Release) release=nullptr;
    decltype(&AI_ShutdownWorker) shutdown=nullptr;
    bool close_worker=false;
    ~ModelGuard(){if(handle &&release)release(handle);if(close_worker &&shutdown)shutdown();}
};
}

int wmain(int argc,wchar_t** argv) {
    try {
        std::map<std::wstring,std::filesystem::path> args;
        bool preserve_worker=false;
        for(int i=1;i<argc;i+=2) { require(i+1<argc,"Every option needs a value"); if(std::wstring(argv[i])==L"--preserve-worker")preserve_worker=std::wstring(argv[i+1])==L"1";else args[argv[i]]=std::filesystem::absolute(argv[i+1]); }
        for(const auto* key:{L"--dll",L"--header",L"--model",L"--image",L"--report"})require(args.count(key)!=0,"Required: --dll --header --model --image --report");
        const auto directory=args[L"--dll"].parent_path();
        Library library; library.value=LoadLibraryExW(args[L"--dll"].c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(library.value!=nullptr,"Cannot load requested DLL; Win32="+std::to_string(GetLastError()));
        wchar_t actual_dll[32768]{}; GetModuleFileNameW(library.value,actual_dll,32768);
        require(std::filesystem::equivalent(actual_dll,args[L"--dll"]),"Loaded a different DLL");
        const auto header_bytes=bytes(args[L"--header"]);
        const std::string header(header_bytes.begin(),header_bytes.end());
        const std::regex pattern(R"(AIENGINE_EXPORT[\s\S]*?AIENGINE_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*\()");
        std::set<std::string> names;
        for(std::sregex_iterator i(header.begin(),header.end(),pattern),end;i!=end;++i)names.insert((*i)[1]);
        require(names.size()==60,"Header must describe exactly 60 public functions");
        for(const auto& name:names)require(GetProcAddress(library.value,name.c_str())!=nullptr,"Missing export "+name);
        auto version=function<decltype(&AI_GetVersion)>(library.value,"AI_GetVersion");
        auto error=function<decltype(&AI_GetLastError)>(library.value,"AI_GetLastError");
        auto shutdown=function<decltype(&AI_ShutdownWorker)>(library.value,"AI_ShutdownWorker");
        auto create=function<decltype(&YOLO_Create)>(library.value,"YOLO_Create");
        auto load=function<decltype(&YOLO_LoadModelFromPath)>(library.value,"YOLO_LoadModelFromPath");
        using Infer = const char* (AIENGINE_CALL *)(int32_t,const uint8_t*,int32_t,float,int32_t,int32_t);
        auto infer=function<Infer>(library.value,"YOLO_InferJson");
        auto status=function<decltype(&YOLO_GetRuntimeStatusJson)>(library.value,"YOLO_GetRuntimeStatusJson");
        auto global_status=function<decltype(&AI_GetRuntimeStatusJson)>(library.value,"AI_GetRuntimeStatusJson");
        auto release=function<decltype(&YOLO_Release)>(library.value,"YOLO_Release");
        ModelGuard guard; guard.release=release; guard.shutdown=shutdown;
        require(std::string(version())=="CQ_X86/" AIENGINE_VERSION_STRING,"DLL version differs from public header");
        SetEnvironmentVariableW(L"CQ_AI_YOLO_CPU_THREADS",L"1");
        SetEnvironmentVariableW(L"CQ_AI_YOLO_PRECISION",L"fp32");
        SetEnvironmentVariableW(L"CQ_AI_YOLO_CUDA_GRAPH",L"0");
        SetEnvironmentVariableW(L"CQ_AI_YOLO_CALIBRATION_FILE",nullptr);
        int32_t handle=0; require(create(&handle)==0,error());
        guard.handle=handle;
        const auto model=narrow(args[L"--model"].wstring(),CP_ACP);
        require(load(handle,model.c_str(),nullptr,0,AI_DEVICE_CPU,0,5)==0,error());
        char runtime[16384]{}; require(status(handle,runtime,sizeof(runtime))>=0,error());
        char worker_runtime[16384]{}; require(global_status(worker_runtime,sizeof(worker_runtime))>=0,error());
        const HANDLE pipe=CreateFileA(ai_worker::pipe_name(ai_worker::RuntimeFlavor::Core),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_EXISTING,0,nullptr);
        require(pipe!=INVALID_HANDLE_VALUE,"Cannot inspect running Worker pipe");
        const uint64_t request_id=(static_cast<uint64_t>(GetCurrentProcessId())<<32)|0xf001;
        const ai_worker::Header request{ai_worker::kMagic,ai_worker::kVersion,ai_worker::CMD_RUNTIME_STATUS,0,request_id};
        require(ai_worker::write_frame(pipe,&request,sizeof(request)),"Cannot send delivery protocol probe");
        ai_worker::ResponseHeader reply{};
        require(ai_worker::read_frame(pipe,&reply,sizeof(reply)) &&reply.magic==ai_worker::kMagic &&reply.version==26 &&reply.request_id==request_id &&reply.status==AI_OK &&reply.payload_size<=ai_worker::kMaxPayload,"Worker protocol or request ID differs from delivery");
        std::vector<uint8_t> payload(reply.payload_size);
        require(ai_worker::read_frame(pipe,payload.data(),reply.payload_size),"Incomplete delivery protocol response");
        ULONG pid=0; const BOOL has_pid=GetNamedPipeServerProcessId(pipe,&pid); CloseHandle(pipe); require(has_pid!=FALSE,"Cannot get Worker PID");
        const HANDLE process=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid); require(process!=nullptr,"Cannot inspect Worker process");
        wchar_t worker[32768]{}; DWORD size=32768; const BOOL has_path=QueryFullProcessImageNameW(process,0,worker,&size); CloseHandle(process);
        require(has_path!=FALSE &&std::filesystem::equivalent(worker,directory/L"CQ_AI_worker.exe"),"DLL used a Worker from a different directory");
        guard.close_worker=!preserve_worker;
        const auto frame=bytes(args[L"--image"]);
        std::string reference=infer(handle,frame.data(),static_cast<int32_t>(frame.size()),.5f,0,0);
        require(!reference.empty() &&reference.front()=='[' &&reference.back()==']',error());
        if(args.count(L"--expected-json")) {
            const auto expected_bytes=bytes(args[L"--expected-json"]); std::string expected(expected_bytes.begin(),expected_bytes.end());
            while(!expected.empty() &&std::isspace(static_cast<unsigned char>(expected.back())))expected.pop_back();
            require(reference==expected,"CPU detections differ from frozen reference");
        }
        std::atomic<int> failures{0}; std::vector<std::thread> callers;
        for(int lane=0;lane<5;++lane)callers.emplace_back([&]{for(int n=0;n<25;++n){const char* result=infer(handle,frame.data(),static_cast<int32_t>(frame.size()),.5f,0,0);if(!result ||reference!=result)++failures;}});
        for(auto& caller:callers)caller.join(); require(failures==0,"Concurrent delivery inference failed or mismatched");
        require(release(handle)==0,error());
        guard.handle=0;
        if(!preserve_worker){require(shutdown()==0,error());guard.close_worker=false;}
        const auto dll_bytes=bytes(args[L"--dll"]),worker_bytes=bytes(directory/L"CQ_AI_worker.exe");
        std::ofstream report(args[L"--report"],std::ios::binary); require(bool(report),"Cannot write report");
        report<<"{\"version\":"<<quoted(version())<<",\"dll\":"<<quoted(narrow(actual_dll))<<",\"worker\":"<<quoted(narrow(worker))
              <<",\"worker_pid\":"<<pid<<",\"worker_protocol\":26,\"public_exports\":60,\"concurrent_calls\":125,\"errors\":0,\"dll_sha256\":"
              <<quoted(ai::sha256(dll_bytes.data(),dll_bytes.size()))<<",\"worker_sha256\":"<<quoted(ai::sha256(worker_bytes.data(),worker_bytes.size()))
              <<",\"detections\":"<<reference<<",\"runtime\":"<<runtime<<",\"worker_runtime\":"<<worker_runtime<<"}\n";
        std::cout<<"Delivery verified: "<<version()<<", requested DLL and paired Worker, 60 exports, 125 concurrent calls\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
