#define NOMINMAX
#include "ai_engine.h"
#include "sha256.h"
#include "worker_protocol.h"
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace {
std::vector<uint8_t> bytes(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot read "+path.u8string());
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
std::string quote(const std::string& value) {
    std::string output="\"";
    for(unsigned char c:value){switch(c){case '"':output+="\\\"";break;case '\\':output+="\\\\";break;case '\n':output+="\\n";break;case '\r':output+="\\r";break;case '\t':output+="\\t";break;default:if(c>=32)output+=c;}}
    return output+'"';
}
int64_t qpc(){LARGE_INTEGER value{};QueryPerformanceCounter(&value);return value.QuadPart;}
void set_environment(const wchar_t* name,const std::string& utf8) {
    const int size=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,utf8.data(),static_cast<int>(utf8.size()),nullptr,0);
    std::wstring value(size,0);
    if(size)MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,utf8.data(),static_cast<int>(utf8.size()),value.data(),size);
    if((!utf8.empty() &&!size) ||!SetEnvironmentVariableW(name,value.c_str()))throw std::runtime_error("Cannot configure benchmark environment");
}
class Barrier {
    std::mutex mutex;std::condition_variable cv;int total,count=0,generation=0;
public:
    explicit Barrier(int n):total(n){}
    void wait(){std::unique_lock<std::mutex> lock(mutex);const int old=generation;
        if(++count==total){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=old;});}
};
struct Resources { uint64_t cpu=0,private_bytes=0,working_set=0; };
uint64_t integer_time(FILETIME t){return (static_cast<uint64_t>(t.dwHighDateTime)<<32)|t.dwLowDateTime;}
Resources resources(HANDLE process) {
    Resources r;if(!process)return r;FILETIME create{},exit{},kernel{},user{};
    if(GetProcessTimes(process,&create,&exit,&kernel,&user))r.cpu=integer_time(kernel)+integer_time(user);
    PROCESS_MEMORY_COUNTERS_EX memory{};memory.cb=sizeof(memory);
    if(GetProcessMemoryInfo(process,reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),sizeof(memory))) {r.private_bytes=memory.PrivateUsage;r.working_set=memory.WorkingSetSize;}
    return r;
}
std::string stats(std::vector<double> samples) {
    std::sort(samples.begin(),samples.end());
    if(samples.empty())return "{}";
    const auto percentile=[&](double p){const size_t index=static_cast<size_t>(std::ceil(p*samples.size()))-1;return samples[std::min(index,samples.size()-1)];};
    std::ostringstream output;output<<std::setprecision(9)<<"{\"count\":"<<samples.size()<<",\"p50_ms\":"<<percentile(.5)<<",\"p95_ms\":"<<percentile(.95)<<",\"p99_ms\":"<<percentile(.99)<<'}';return output.str();
}
}
int wmain(int argc,wchar_t** argv) {
    int32_t handle=0;
    bool shutdown_worker=true;
    try {
        std::filesystem::path model,output="yolo-benchmark.json",verification;
        std::vector<std::filesystem::path> images;
        int device=2,sessions=5,callers=5,warmup=100,samples=1000,rounds=3,threads=0,seconds=0,worker_protocol=26;
        float conf=.5f;bool burst=false;std::string precision="fp32",workload,profile;int graph=0;
        const auto utf8=[](const wchar_t* value){const int n=WideCharToMultiByte(CP_UTF8,0,value,-1,nullptr,0,nullptr,nullptr);std::string text(n,0);WideCharToMultiByte(CP_UTF8,0,value,-1,text.data(),n,nullptr,nullptr);text.pop_back();return text;};
        for(int i=1;i<argc;++i) {
            const std::wstring key=argv[i];if(i+1>=argc)throw std::runtime_error("Every option requires a value");const wchar_t* value=argv[++i];
            if(key==L"--model")model=value;else if(key==L"--image")images.emplace_back(value);
            else if(key==L"--images-dir"){for(const auto& entry:std::filesystem::directory_iterator(value))if(entry.path().extension()==L".bmp")images.push_back(entry.path());}
            else if(key==L"--output")output=value;else if(key==L"--verify-output")verification=value;
            else if(key==L"--device")device=std::stoi(value);else if(key==L"--sessions")sessions=std::stoi(value);
            else if(key==L"--callers")callers=std::stoi(value);else if(key==L"--threads")threads=std::stoi(value);
            else if(key==L"--warmup")warmup=std::stoi(value);else if(key==L"--samples")samples=std::stoi(value);
            else if(key==L"--rounds")rounds=std::stoi(value);else if(key==L"--seconds")seconds=std::stoi(value);
            else if(key==L"--worker-protocol")worker_protocol=std::stoi(value);
            else if(key==L"--shutdown-worker")shutdown_worker=std::stoi(value)!=0;
            else if(key==L"--mode")burst=std::wstring(value)==L"simultaneous";
            else if(key==L"--precision")precision=utf8(value);else if(key==L"--graph")graph=std::stoi(value);
            else if(key==L"--confidence")conf=std::stof(value);else if(key==L"--workload")workload=utf8(value);
            else if(key==L"--profile")profile=utf8(value);else throw std::runtime_error("Unknown option: "+utf8(key.c_str()));
        }
        if(model.empty() ||images.empty() ||callers<=0 ||callers>64 ||sessions<=0 ||warmup<0 ||samples<1 ||rounds<1 ||seconds<0)throw std::runtime_error("Required: --model path --image BMP; positive callers/sessions/samples/rounds");
        if(seconds &&burst)throw std::runtime_error("Duration mode requires --mode continuous");
        if(worker_protocol!=25 &&worker_protocol!=26)throw std::runtime_error("Benchmark supports Worker protocols 25 and 26");
        std::sort(images.begin(),images.end());std::vector<std::vector<uint8_t>> frames;
        std::string image_hashes;
        for(const auto& image:images){frames.push_back(bytes(image));image_hashes+=ai::sha256(frames.back().data(),frames.back().size());}
        if(workload.empty())workload=ai::sha256(image_hashes.data(),image_hashes.size());
        set_environment(L"CQ_AI_YOLO_CPU_THREADS",std::to_string(threads));
        set_environment(L"CQ_AI_YOLO_PRECISION",precision);set_environment(L"CQ_AI_YOLO_CUDA_GRAPH",std::to_string(graph));
        set_environment(L"CQ_AI_YOLO_WORKLOAD_ID",workload);set_environment(L"CQ_AI_YOLO_CALIBRATION_FILE",profile);
        set_environment(L"CQ_AI_YOLO_VALIDATING",verification.empty()?"0":"1");
        LARGE_INTEGER frequency{};QueryPerformanceFrequency(&frequency);
        const auto cold_start=qpc();
        if(YOLO_Create(&handle)<0)throw std::runtime_error(AI_GetLastError());
        const auto load_start=qpc();const auto model_data=bytes(model);
        const int loaded=YOLO_LoadModelFromMemory(handle,model_data.data(),static_cast<int32_t>(model_data.size()),nullptr,0,0,device,0,sessions);
        const auto load_end=qpc();if(loaded<0)throw std::runtime_error(AI_GetLastError());
        char status_buffer[16384]{};
        if(YOLO_GetRuntimeStatusJson(handle,status_buffer,sizeof(status_buffer))<0)throw std::runtime_error(AI_GetLastError());
        const std::string runtime=status_buffer;
        std::vector<std::string> expected_results;
        if(!verification.empty() ||seconds){
            expected_results.resize(frames.size());
            std::ofstream file;
            if(!verification.empty()){file.open(verification);file<<'[';}
            for(size_t i=0;i<frames.size();++i){const auto& frame=frames[i];const char* result=YOLO_InferJson(handle,frame.data(),static_cast<int32_t>(frame.size()),conf,0,0);
                if(!result ||!*result)throw std::runtime_error(AI_GetLastError());
                expected_results[i]=result;
                if(file.is_open()){if(i)file<<',';file<<"{\"image\":"<<quote(images[i].u8string())<<",\"sha256\":"<<quote(ai::sha256(frame.data(),frame.size()))<<",\"detections\":"<<result<<'}';}}
            if(file.is_open()){file<<']';file.flush();if(!file)throw std::runtime_error("Cannot write verification output");}
        }
        // An imported function's address may be an EXE thunk. Resolve the
        // loaded library by name so the recorded hash identifies the product.
#if defined(_M_IX86)
        HMODULE module=GetModuleHandleW(L"CQ_X86.dll");
#else
        HMODULE module=GetModuleHandleW(L"CQ_AI_x64.dll");
#endif
        if(!module)throw std::runtime_error("Cannot identify loaded business DLL");
        wchar_t filename[32768]{};GetModuleFileNameW(module,filename,32768);const std::filesystem::path dll=filename;
        GetModuleFileNameW(nullptr,filename,32768);const std::filesystem::path executable=filename;
        HANDLE worker=nullptr;std::filesystem::path worker_path;
#if defined(_M_IX86)
        // Protocol 25 is supported only to measure the preserved old DLL.
        // Product routing never uses this benchmark-only legacy pipe name.
        const char* pipe_name=worker_protocol==25?R"(\\.\pipe\cq_ai_worker_v23_core_0145)":ai_worker::pipe_name(ai_worker::RuntimeFlavor::Core);
        HANDLE pipe=CreateFileA(pipe_name,GENERIC_READ | GENERIC_WRITE,0,nullptr,OPEN_EXISTING,0,nullptr);
        ULONG pid=0;if(pipe!=INVALID_HANDLE_VALUE){GetNamedPipeServerProcessId(pipe,&pid);CloseHandle(pipe);}
        if(pid){worker=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,FALSE,pid);DWORD length=32768;if(worker &&QueryFullProcessImageNameW(worker,0,filename,&length))worker_path=filename;}
#endif
        const auto dll_data=bytes(dll),exe_data=bytes(executable);
        std::ostringstream report;report<<std::setprecision(9)<<"{\"schema\":1,\"qpc_frequency\":"<<frequency.QuadPart<<",\"api_version\":"<<quote(AI_GetVersion())
            <<",\"benchmark_protocol\":"<<ai_worker::kVersion<<",\"worker_protocol\":"
#if defined(_M_IX86)
            <<worker_protocol
#else
            <<"null"
#endif
            <<",\"model_sha256\":"<<quote(ai::sha256(model_data.data(),model_data.size()))
            <<",\"dll\":"<<quote(dll.u8string())<<",\"dll_sha256\":"<<quote(ai::sha256(dll_data.data(),dll_data.size()))
            <<",\"benchmark_sha256\":"<<quote(ai::sha256(exe_data.data(),exe_data.size()))
            <<",\"workload\":"<<quote(workload)<<",\"callers\":"<<callers<<",\"mode\":"<<quote(burst?"simultaneous":"continuous")
            <<",\"confidence\":"<<conf<<",\"nms\":0.45,\"warmup_per_lane\":"<<warmup<<",\"samples_per_lane\":"<<samples
            <<",\"cold_create_ms\":"<<(load_start-cold_start)*1000.0/frequency.QuadPart<<",\"load_ms\":"<<(load_end-load_start)*1000.0/frequency.QuadPart<<",\"runtime\":"<<runtime;
        if(!worker_path.empty()){const auto data=bytes(worker_path);report<<",\"worker\":"<<quote(worker_path.u8string())<<",\"worker_sha256\":"<<quote(ai::sha256(data.data(),data.size()));}
        report<<",\"rounds\":[";
        std::ofstream raw(output.wstring()+L".samples.csv");raw<<"round,lane,sample,frame,total_ms\n";
        std::ofstream memory_file(output.wstring()+L".resources.csv");memory_file<<"round,elapsed_s,client_private,worker_private,client_working_set,worker_working_set\n";
        for(int round=0;round<rounds;++round) {
            Barrier barrier(callers);std::vector<std::vector<double>> timings(callers);
            // Commit sample storage before the soak starts, so benchmark
            // vector growth is not misreported as engine memory growth.
            for(auto& lane:timings){lane.resize(seconds?static_cast<size_t>(seconds)*1000:samples);std::fill(lane.begin(),lane.end(),0.0);lane.clear();}
            std::atomic<int> errors{0};std::vector<std::thread> callers_threads;std::atomic<bool> monitor_stop{false};
            uint64_t client_peak=0,worker_peak=0;const auto resource_start=qpc();
            std::thread monitor([&]{while(!monitor_stop){const auto client=resources(GetCurrentProcess()),remote=resources(worker);client_peak=std::max(client_peak,client.private_bytes);worker_peak=std::max(worker_peak,remote.private_bytes);
                memory_file<<round<<','<<(qpc()-resource_start)/static_cast<double>(frequency.QuadPart)<<','<<client.private_bytes<<','<<remote.private_bytes<<','<<client.working_set<<','<<remote.working_set<<'\n';
                for(int i=0;i<10 &&!monitor_stop;++i)Sleep(100);}});
            Resources before_client,before_worker;
            int64_t begin=0,end=0;
            for(int lane=0;lane<callers;++lane)callers_threads.emplace_back([&,lane] {
                for(int n=0;n<warmup;++n){const auto& frame=frames[(n+lane)%frames.size()];if(!*YOLO_InferJson(handle,frame.data(),static_cast<int32_t>(frame.size()),conf,0,0))++errors;}
                barrier.wait();if(lane==0){before_client=resources(GetCurrentProcess());before_worker=resources(worker);begin=qpc();}barrier.wait();
                int n=0;const auto deadline=qpc()+static_cast<int64_t>(seconds)*frequency.QuadPart;
                while(seconds?qpc()<deadline:n<samples) {
                    if(burst)barrier.wait();const auto& frame=frames[(n+lane)%frames.size()];const auto start=qpc();
                    const char* result=YOLO_InferJson(handle,frame.data(),static_cast<int32_t>(frame.size()),conf,0,0);const auto stop=qpc();
                    if(!result ||!*result ||(!expected_results.empty() &&expected_results[(n+lane)%frames.size()]!=result))++errors;
                    timings[lane].push_back((stop-start)*1000.0/frequency.QuadPart);
                    if(burst)barrier.wait();++n;
                }
                barrier.wait();if(lane==0)end=qpc();
            });
            for(auto& thread:callers_threads)thread.join();monitor_stop=true;monitor.join();
            const auto after_client=resources(GetCurrentProcess()),after_worker=resources(worker);
            const double elapsed=(end-begin)/static_cast<double>(frequency.QuadPart);
            if(round)report<<',';report<<"{\"round\":"<<round<<",\"errors\":"<<errors.load()<<",\"elapsed_s\":"<<elapsed<<",\"lanes\":[";
            std::vector<double> all;
            for(int lane=0;lane<callers;++lane){if(lane)report<<',';report<<stats(timings[lane]);all.insert(all.end(),timings[lane].begin(),timings[lane].end());
                for(size_t n=0;n<timings[lane].size();++n)raw<<round<<','<<lane<<','<<n<<','<<(n+lane)%frames.size()<<','<<timings[lane][n]<<'\n';}
            report<<"],\"overall\":"<<stats(all)<<",\"throughput_per_s\":"<<all.size()/elapsed
                <<",\"client_cpu_core_equivalents\":"<<(after_client.cpu-before_client.cpu)/1e7/elapsed<<",\"worker_cpu_core_equivalents\":"<<(after_worker.cpu-before_worker.cpu)/1e7/elapsed
                <<",\"client_private_peak\":"<<client_peak<<",\"worker_private_peak\":"<<worker_peak<<'}';
            std::cout<<"round="<<round<<" errors="<<errors<<" "<<stats(all)<<" throughput="<<all.size()/elapsed<<std::endl;
            if(errors)throw std::runtime_error("Inference failed during measurement");
        }
        report<<"]}";std::ofstream file(output);file<<report.str();file.flush();if(!file)throw std::runtime_error("Cannot write benchmark report");
        YOLO_Release(handle);handle=0;if(worker)CloseHandle(worker);if(shutdown_worker)AI_ShutdownWorker();return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;if(handle)YOLO_Release(handle);if(shutdown_worker)AI_ShutdownWorker();return 2;}
}
