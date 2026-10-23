#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define AIENGINE_EXPORT extern "C"
#include "ai_engine.h"
#include "sha256.h"
#include "cv_test_hooks.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

using Bytes=std::vector<uint8_t>;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
Bytes read(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);require(bool(f),"Cannot read fixture");return {std::istreambuf_iterator<char>(f),{}};}
void write(const std::filesystem::path& p,const Bytes& b){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),b.size());require(bool(f),"Cannot write owned fixture");}
struct Bmp {
    Bytes bytes;int w,h,stride,offset;bool bottom;
    explicit Bmp(Bytes b):bytes(std::move(b)){
        require(bytes.size()>=54&&bytes[0]=='B'&&bytes[1]=='M'&&bytes[28]==24,"Expected24bit BMP");
        int signed_h;memcpy(&w,bytes.data()+18,4);memcpy(&signed_h,bytes.data()+22,4);memcpy(&offset,bytes.data()+10,4);
        require(w>0&&signed_h!=0&&signed_h!=INT32_MIN,"Invalid geometry");bottom=signed_h>0;h=abs(signed_h);stride=(w*3+3)&~3;
        require(offset>=54&&size_t(offset)+size_t(stride)*h<=bytes.size(),"Invalid BMP bounds");
    }
    uint8_t* row(int y){return bytes.data()+offset+(bottom?h-1-y:y)*stride;}
    const uint8_t* row(int y)const{return bytes.data()+offset+(bottom?h-1-y:y)*stride;}
};
bool visible(const uint8_t* p){return p[0]!=255||p[1]!=0||p[2]!=255;}
void paste(Bmp& dst,const Bmp& t,int x,int y){for(int v=0;v<t.h;++v)for(int u=0;u<t.w;++u)if(visible(t.row(v)+u*3))memcpy(dst.row(y+v)+(x+u)*3,t.row(v)+u*3,3);}
template<class T>T fn(HMODULE m,const char*n){auto p=GetProcAddress(m,n);require(p!=nullptr,"Missing export");return reinterpret_cast<T>(p);}
struct Api {
    using One=int32_t(AIENGINE_CALL*)(int32_t,const char*,const uint8_t*,int32_t,float,int32_t,const char*,CVMatchResult*,int32_t,int32_t);
    using Multi=const char*(AIENGINE_CALL*)(int32_t,const char*,const uint8_t*,int32_t,const char*,float,const char*,int32_t,int32_t);
    HMODULE dll=nullptr;int32_t handle=0;One one;Multi multi;
    decltype(&CV_Release) release;decltype(&AI_GetLastError) error;
    Api(const std::filesystem::path& p,const std::filesystem::path& templates){
        AddDllDirectory(p.parent_path().c_str());dll=LoadLibraryExW(p.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        require(dll!=nullptr,"Cannot load DLL");auto create=fn<decltype(&CV_Create)>(dll,"CV_Create");auto load=fn<decltype(&CV_LoadTemplateDir)>(dll,"CV_LoadTemplateDir");
        release=fn<decltype(&CV_Release)>(dll,"CV_Release");error=fn<decltype(&AI_GetLastError)>(dll,"AI_GetLastError");one=fn<One>(dll,"CV_FindTransparentOne");multi=fn<Multi>(dll,"CV_FindTransparentMultiText");
        require(create(&handle)==0,"CV_Create failed");require(load(handle,templates.string().c_str(),0)>0,"Template loading failed");
    }
    ~Api(){if(handle)release(handle);if(dll)FreeLibrary(dll);}
};
struct Case {std::string name,templ;std::vector<Bmp> frames;std::vector<std::pair<int,int>> expected;int mode=0;bool multi=false;};
void check(Api& api,const Case& c,size_t index,float threshold=.9f,double* latency=nullptr){
    const auto& b=c.frames[index%c.frames.size()];const auto xy=c.expected[index%c.expected.size()];
    LARGE_INTEGER begin{},end{};static const double frequency=[]{LARGE_INTEGER f{};QueryPerformanceFrequency(&f);return double(f.QuadPart);}();
    if(c.multi){if(latency)QueryPerformanceCounter(&begin);const char* p=api.multi(api.handle,c.templ.c_str(),b.bytes.data(),int(b.bytes.size()),"",threshold,"FF00FF",0,0);if(latency){QueryPerformanceCounter(&end);*latency=1000.0*(end.QuadPart-begin.QuadPart)/frequency;}require(p!=nullptr,"Null multi result");
        const auto expected=xy.first<0?std::string():"0,"+std::to_string(xy.first)+","+std::to_string(xy.second);require(p==expected,"Multi result mismatch");
    }else{CVMatchResult r{};if(latency)QueryPerformanceCounter(&begin);const int n=api.one(api.handle,c.templ.c_str(),b.bytes.data(),int(b.bytes.size()),threshold,c.mode,"FF00FF",&r,0,0);if(latency){QueryPerformanceCounter(&end);*latency=1000.0*(end.QuadPart-begin.QuadPart)/frequency;}
        require(n==(xy.first<0?0:1),"Single status mismatch");if(n>0){require(r.x==xy.first&&r.y==xy.second,"Single position mismatch");
            if(c.name.rfind("new_",0)==0){require(r.w==95&&r.h==81,"Single dimensions mismatch");require(std::isfinite(r.sim)&&r.sim>=threshold,"Single score mismatch");if(c.mode==0)require(r.sim==1.f,"Identical visible pixels must score1");}}}
    require(api.error()[0]==0,"Last error is nonempty");
}
std::string quoted(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';o+=c;}return o+'\"';}
int wmain(int argc,wchar_t**argv){try{
    std::map<std::wstring,std::wstring>a;for(int i=1;i+1<argc;i+=2)a[argv[i]]=argv[i+1];
    require(a.count(L"--dll")&&a.count(L"--input")&&a.count(L"--output"),"Required --dll --input --output");
    const auto dll=std::filesystem::absolute(a[L"--dll"]),root=std::filesystem::absolute(a[L"--input"]),out=std::filesystem::absolute(a[L"--output"]);
    const int samples=a.count(L"--samples")?std::stoi(a[L"--samples"]):1000,rounds=a.count(L"--rounds")?std::stoi(a[L"--rounds"]):3;
    const std::wstring scope=a.count(L"--scope")?a[L"--scope"]:L"all";
    const auto temp=out.parent_path()/("owned-templates-"+std::to_string(GetCurrentProcessId()));std::filesystem::create_directories(temp);
    struct Guard{std::filesystem::path p;~Guard(){std::error_code ec;std::filesystem::remove_all(p,ec);}} guard{temp};
    std::vector<Case> cases;
    if(scope!=L"old"){
        Bmp t(read(root/L"1.bmp")),b(read(root/L"2.bmp"));write(temp/"new.bmp",t.bytes);
        Case fixed{"new_color","new.bmp",{b},{{596,165}},0,false};cases.push_back(fixed);
        auto gray=fixed;gray.name="new_gray";gray.mode=1;cases.push_back(gray);
        auto multi=fixed;multi.name="new_multi";multi.multi=true;cases.push_back(multi);
        Case changed{"new_changed_color","new.bmp",{}, {},0,false};
        uint32_t state=0x6755;
        for(int i=0;i<32;++i){Bmp f=b;auto step=[&]{state=1664525u*state+1013904223u;return uint8_t(state>>24);};
            for(int y=0;y<f.h;++y)for(int x=0;x<f.w*3;++x)f.row(y)[x]=step();
            const int x=20+(i*31)%(b.w-t.w-40),y=20+(i*17)%(b.h-t.h-40);paste(f,t,x,y);changed.frames.push_back(std::move(f));changed.expected.emplace_back(x,y);
        }
        cases.push_back(changed);auto changed_multi=changed;changed_multi.name="new_changed_multi";changed_multi.multi=true;cases.push_back(std::move(changed_multi));
        auto changed_gray=changed;changed_gray.name="new_changed_gray";changed_gray.mode=1;cases.push_back(std::move(changed_gray));
        Bmp blank=b;for(int y=0;y<blank.h;++y)memset(blank.row(y),0,blank.w*3);
        cases.push_back({"new_miss","new.bmp",{blank},{{-1,-1}},0,false});
        cases.push_back({"new_gray_miss","new.bmp",{blank},{{-1,-1}},1,false});
        cases.push_back({"new_multi_miss","new.bmp",{blank},{{-1,-1}},0,true});
    }
    if(scope!=L"new"){
        Bmp t(read(root/L"xyq.bmp")),m(read(root/L"CV_123.bmp")),b(read(root/std::filesystem::u8path(u8"CV_大图.bmp")));write(temp/"old.bmp",t.bytes);write(temp/"miss.bmp",m.bytes);
        Case hit{"old_fixed_hit","old.bmp",{b},{{324,105}},0,false};cases.push_back(hit);auto miss=hit;miss.name="old_fixed_miss";miss.templ="miss.bmp";miss.expected={{-1,-1}};cases.push_back(miss);
        auto changed=hit;changed.name="old_changed_hit";changed.frames.clear();changed.expected.clear();
        for(int i=0;i<32;++i){Bmp f=b;for(int y=0;y<f.h;++y)for(int x=0;x<f.w;++x){if(x>=324&&x<324+t.w&&y>=105&&y<105+t.h)continue;if(((x*13+y*7+i*11)&63)==0)f.row(y)[x*3+i%3]^=uint8_t(i+1);}
            int px=324,py=105;if(i%2){for(int y=0;y<t.h;++y)memcpy(f.row(105+y)+324*3,f.row(180+y)+600*3,t.w*3);px=30+(i*17)%250;py=30+(i*13)%100;paste(f,t,px,py);}changed.frames.push_back(std::move(f));changed.expected.emplace_back(px,py);}
        cases.push_back(changed);changed.name="old_changed_miss";changed.templ="miss.bmp";changed.expected.assign(32,{-1,-1});cases.push_back(std::move(changed));
    }
    Api api(dll,temp);
    std::ofstream report(out),raw(out.wstring()+L".samples.csv");raw<<"case,round,sample,total_ms\n";
    const auto dll_bytes=read(dll);
    const auto small_bytes=read(root/L"1.bmp"),big_bytes=read(root/L"2.bmp");
    report<<std::setprecision(10)<<"{\"schema\":1,\"architecture\":"<<quoted(sizeof(void*)==4?"x86":"x64")<<",\"test_hooks\":"<<(GetProcAddress(api.dll,"CVTest_Stats")?"true":"false")<<",\"input_sha256\":{\"1.bmp\":"<<quoted(ai::sha256(small_bytes.data(),small_bytes.size()))<<",\"2.bmp\":"<<quoted(ai::sha256(big_bytes.data(),big_bytes.size()))<<"},\"dll_sha256\":"<<quoted(ai::sha256(dll_bytes.data(),dll_bytes.size()))<<",\"warmup\":64,\"samples\":"<<samples<<",\"rounds\":"<<rounds<<",\"limit_ms\":5,\"cases\":[";
    bool all=true;int ci=0;LARGE_INTEGER frequency{};QueryPerformanceFrequency(&frequency);
    for(const auto& c:cases){double cold_ms;check(api,c,0,.9f,&cold_ms);for(size_t i=0;i<c.frames.size();++i)check(api,c,i);
        if(c.name=="new_color"||c.name=="new_multi")for(float t:{.9f,.99f,1.f})check(api,c,0,t);
        if(c.name=="new_gray")check(api,c,0,.99f);
        if(ci++)report<<',';report<<"{\"name\":"<<quoted(c.name)<<",\"cold_ms\":"<<cold_ms<<",\"rounds\":[";
        for(int r=0;r<rounds;++r){for(int k=0;k<64;++k)check(api,c,k);
            std::vector<double> times;times.reserve(samples);
            for(int k=0;k<samples;++k){double ms;check(api,c,k,.9f,&ms);times.push_back(ms);}
            for(int k=0;k<samples;++k)raw<<c.name<<','<<r<<','<<k<<','<<times[k]<<'\n';std::sort(times.begin(),times.end());
            auto percentile=[&](double q){return times[size_t(std::ceil(q*samples))-1];};const double p95=percentile(.95);const auto over=times.end()-std::upper_bound(times.begin(),times.end(),5.);
            all&=p95<=5.;if(r)report<<',';report<<"{\"count\":"<<samples<<",\"errors\":0,\"p50_ms\":"<<percentile(.5)<<",\"p95_ms\":"<<p95<<",\"p99_ms\":"<<percentile(.99)<<",\"max_ms\":"<<times.back()<<",\"over5_fraction\":"<<double(over)/samples<<'}';report.flush();
            std::cout<<c.name<<" round="<<r<<" p95="<<p95<<" max="<<times.back()<<" ms\n";
            const auto profile=reinterpret_cast<void(*)(CVTestTransparentProfile*)>(GetProcAddress(api.dll,"CVTest_TransparentProfile"));
            if(profile){CVTestTransparentProfile p{};profile(&p);std::cout<<"  bounds="<<p.rough_ms<<" exact="<<p.exact_ms<<" candidates="<<p.candidates<<'\n';}
        }report<<"]}";report.flush();
    }report<<"],\"accepted_5ms\":"<<(all&&samples>=1000&&rounds==3?"true":"false")<<"}\n";
    return all?0:3;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
