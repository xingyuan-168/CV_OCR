#include "ai_engine.h"
#if defined(AIENGINE_CV_TEST_HOOKS)
#include "cv_test_hooks.h"
#endif
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

using Bytes = std::vector<uint8_t>;
void u32(Bytes& b, size_t p, uint32_t n) { for (int i=0; i<4; ++i) b[p+i]=uint8_t(n>>(8*i)); }
Bytes bmp(int w, int h, uint32_t seed) {
    const int stride=(w*3+3)&~3;
    Bytes b(54+stride*h);
    b[0]='B'; b[1]='M'; u32(b,2,uint32_t(b.size())); u32(b,10,54); u32(b,14,40);
    u32(b,18,w); u32(b,22,uint32_t(-h)); b[26]=1; b[28]=24;
    for (size_t i=54;i<b.size();++i) { seed=seed*1664525u+1013904223u; b[i]=uint8_t(seed>>24); }
    return b;
}
void paste(Bytes& dst, int dw, const Bytes& src, int sw, int sh, int x, int y) {
    for (int row=0;row<sh;++row)
        std::memcpy(dst.data()+54+(row+y)*((dw*3+3)&~3)+x*3,
                    src.data()+54+row*((sw*3+3)&~3),sw*3);
}
struct TempDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cq-cv-stability-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDirectory() { std::filesystem::create_directories(path); }
    ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path,error); }
};
uint64_t private_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),sizeof(counters)))
        return counters.PrivateUsage;
#endif
    return 0;
}
int main(int argc, char** argv) {
#if defined(_WIN32)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    int seconds=0, thread_count=4;
    bool memory_sweep=false;
    for (int i=1;i+1<argc;++i) {
        if (std::string(argv[i])=="--stress-seconds") seconds=std::stoi(argv[++i]);
        else if (std::string(argv[i])=="--threads") thread_count=std::stoi(argv[++i]);
        else if (std::string(argv[i])=="--memory-sweep") memory_sweep=std::stoi(argv[++i])!=0;
    }
    assert(thread_count>0 && thread_count<=64 && seconds>=0);
    TempDirectory directory;
    std::vector<Bytes> templates;
    for (int i=0;i<32;++i) {
        templates.push_back(bmp(8+i,8+i/2,12345+i*99));
        std::ofstream file(directory.path/(std::to_string(i)+".bmp"),std::ios::binary);
        file.write(reinterpret_cast<const char*>(templates.back().data()),templates.back().size());
    }
    int32_t handle=0;
    assert(CV_Create(&handle)==AI_OK);
    assert(CV_LoadTemplateDir(handle,directory.path.u8string().c_str(),0)==32);
    auto frame=bmp(160,96,777);
    paste(frame,160,templates[0],8,8,10,10);
    auto call=[&](int h,const char* names,const char* bias,float score,int mode,const Bytes& data) {
        return std::string(CV_FindMultiText(h,names,data.data(),int32_t(data.size()),bias,score,mode,0,0));
    };
    auto invalid=[&](const char* reason,int h,const char* names,const char* bias,float score,int mode) {
        assert(call(h,names,bias,score,mode,frame).empty());
        const std::string error=AI_GetLastError();
        assert(error.find("CV_FindMultiText:")==0 && error.find(reason)!=std::string::npos);
    };
    invalid("handle",0,"0.bmp","",.99f,0);
    for (const char* names : {"","|0.bmp","0.bmp|","0.bmp||1.bmp"}) invalid("template_names",handle,names,"",.99f,0);
    // Guard against reading offsets +2/+4 before rejecting an invalid mode.
    invalid("match_mode",handle,"0.bmp","A",.99f,2);
    for (const char* bias : {"A","GGGGGG","12345","1234567"}) invalid("color_bias",handle,"0.bmp",bias,.99f,0);
    invalid("color_bias",handle,"0.bmp","123456",.99f,1);
    for (float score : {-1.f,1.1f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
        invalid("min_score",handle,"0.bmp","",score,0);
    invalid("template not found",handle,"missing.bmp","",.99f,0);
    for (int variant=0;variant<4;++variant) {
        Bytes bad=frame;
        if (variant==0) u32(bad,22,0x80000000u);
        if (variant==1) u32(bad,18,0x7fffffffu);
        if (variant==2) u32(bad,14,0xffffffffu);
        if (variant==3) bad.resize(55);
        assert(call(handle,"0.bmp","",.99f,0,bad).empty());
        assert(std::strstr(AI_GetLastError(),"image data")!=nullptr);
    }
    CVMatchResult one{};
    assert(CV_FindOne(handle,"0.bmp",frame.data(),int32_t(frame.size()),.99f,2,&one,0,0)==AI_ERR_INVALID_ARGUMENT);
    assert(std::strstr(AI_GetLastError(),"match_mode"));
    assert(CV_FindTransparentOne(handle,"0.bmp",frame.data(),int32_t(frame.size()),NAN,0,"FF00FF",&one,0,0)==AI_ERR_INVALID_ARGUMENT);
    assert(std::strstr(AI_GetLastError(),"min_score"));
    assert(std::strlen(CV_FindTransparentMultiText(handle,"0.bmp",frame.data(),int32_t(frame.size()),"A",.99f,"FF00FF",0,0))==0);
    assert(std::strstr(AI_GetLastError(),"color_bias"));
    assert(!call(handle,"0.bmp","",.99f,0,frame).empty());
    assert(AI_GetLastError()[0]=='\0');

#if defined(AIENGINE_CV_TEST_HOOKS) && defined(AIENGINE_WITH_OPENCV)
    for (int stage : {1,2,3}) for (int kind : {1,2,3,4}) {
        auto before=CVTest_Stats();
        CVTest_Fault(stage,kind);
        assert(call(handle,"0.bmp","",.99f,0,frame).empty());
        std::string error=AI_GetLastError();
        assert(error.find("CV_FindMultiText:")==0);
        assert(error.find(kind==1 ? "out of memory" : kind==2 ? "OpenCV" : kind==3 ? "standard" : "unknown")!=std::string::npos);
        auto after=CVTest_Stats();
        assert(after.active==0);
        if (stage<3) assert(after.discarded==before.discarded+1);
        assert(!call(handle,"0.bmp","",.99f,0,frame).empty());
        assert(AI_GetLastError()[0]=='\0');
    }
    // Snapshot owners outlive clear/release only until the in-flight call returns.
    for (bool release : {false,true}) {
        CVTest_Pause(1);
        std::thread in_flight([&] { assert(!call(handle,"0.bmp","",.99f,0,frame).empty()); });
        auto timeout=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!CVTest_Paused() && std::chrono::steady_clock::now()<timeout) std::this_thread::yield();
        assert(CVTest_Paused());
        assert((release ? CV_Release(handle) : CV_ClearTemplateCache(handle))==AI_OK);
        assert(CVTest_Stats().live_templates==(release ? 32u : 1u));
        CVTest_Pause(0);
        in_flight.join();
        assert(CVTest_Stats().live_templates==0);
        if(release) assert(CV_Create(&handle)==AI_OK);
        assert(CV_LoadTemplateDir(handle,directory.path.u8string().c_str(),0)==32);
    }
    auto before=CVTest_Stats();
    CVTest_OverBudget();
    auto after=CVTest_Stats();
    assert(after.discarded==before.discarded+1 && after.idle_bytes<=after.budget);
    assert(CV_ClearTemplateCache(handle)==AI_OK);
    assert(CVTest_Stats().live_templates==0);
    assert(CV_LoadTemplateDir(handle,directory.path.u8string().c_str(),0)==32);
#endif

    if(memory_sweep) {
        for (int width : {640,1920,4096}) {
            const int height=width==640 ? 480 : width==1920 ? 1080 : 2160;
            auto large=bmp(width,height,3456+width);
            paste(large,width,templates[0],8,8,10,10);
            assert(call(handle,"0.bmp","",.999f,0,large)=="0,10,10");
#if defined(AIENGINE_CV_TEST_HOOKS)
            auto stats=CVTest_Stats();
            assert(stats.active==0 && stats.idle_bytes<=stats.budget);
#endif
            std::cout<<"memory_sweep="<<width<<"x"<<height<<" private_bytes="<<private_bytes()<<std::endl;
        }
    }

    // 32 sizes x 4 changing frames, two templates per call. Expected output is
    // independently known from placement, then checked against a serial call.
    struct Case { Bytes data; std::string names, expected; };
    std::vector<Case> cases;
    for (int i=0;i<32;++i) for(int v=0;v<4;++v) {
        int j=(i+1)%32;
        Case c{bmp(160,96,10000+i*9+v),std::to_string(i)+".bmp|"+std::to_string(j)+".bmp",""};
        paste(c.data,160,templates[i],8+i,8+i/2,10+v,10);
        paste(c.data,160,templates[j],8+j,8+j/2,90,60+v);
        c.expected=call(handle,c.names.c_str(),"",.999f,0,c.data);
        assert(c.expected=="0,"+std::to_string(10+v)+",10|1,90,"+std::to_string(60+v));
        cases.push_back(std::move(c));
    }
    std::atomic<uint64_t> calls{0};
    std::atomic<bool> go{false};
    std::vector<std::vector<double>> timings(thread_count);
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
    std::vector<std::thread> threads;
    for(int t=0;t<thread_count;++t) threads.emplace_back([&,t] {
        while(!go.load()) std::this_thread::yield();
        size_t iteration=0;
        do {
            const auto& c=cases[(iteration+t*7)%cases.size()];
            auto begin=std::chrono::steady_clock::now();
            assert(call(handle,c.names.c_str(),"",.999f,0,c.data)==c.expected);
            assert(AI_GetLastError()[0]=='\0');
            auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            // Bounded sampling; never let the measurement become a memory leak.
            if(timings[t].size()<10000) timings[t].push_back(elapsed);
            else timings[t][iteration%10000]=elapsed;
            invalid("handle",0,"0.bmp","",.99f,0);
            ++iteration; ++calls;
        } while(seconds ? std::chrono::steady_clock::now()<deadline : iteration<128);
    });
    auto started=std::chrono::steady_clock::now();
    go=true;
    if(seconds) {
        while(std::chrono::steady_clock::now()<deadline) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            std::cout<<"private_bytes="<<private_bytes()<<" calls="<<calls.load()<<std::endl;
#if defined(AIENGINE_CV_TEST_HOOKS)
            auto stats=CVTest_Stats();
            assert(stats.active<=stats.limit && stats.idle_bytes<=stats.budget);
#endif
        }
    }
    for(auto& t:threads)t.join();
    double duration=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::vector<double> samples;
    for(auto& sample:timings) samples.insert(samples.end(),sample.begin(),sample.end());
    std::sort(samples.begin(),samples.end());
    std::cout<<"threads="<<thread_count<<" calls="<<calls<<" seconds="<<duration
             <<" throughput="<<calls/duration<<" p95_ms="<<samples[size_t((samples.size()-1)*.95)]<<std::endl;
    assert(CV_Release(handle)==AI_OK);
#if defined(AIENGINE_CV_TEST_HOOKS)
    auto stats=CVTest_Stats();
    assert(stats.active==0 && stats.idle_bytes<=stats.budget && stats.peak_active<=stats.limit && stats.live_templates==0);
    std::cout<<"pool_limit="<<stats.limit<<" peak_active="<<stats.peak_active<<" idle_bytes="<<stats.idle_bytes<<" budget="<<stats.budget<<std::endl;
#endif
    invalid("handle",handle,"0.bmp","",.99f,0);
    std::cout<<"CV stability PASS"<<std::endl;
}
