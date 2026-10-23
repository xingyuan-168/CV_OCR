#include "ai_engine.h"
#if defined(AIENGINE_CV_TEST_HOOKS)
#include "cv_test_hooks.h"
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>
#include <chrono>

using Bytes=std::vector<uint8_t>;
void put(Bytes& b,int at,int n){memcpy(b.data()+at,&n,4);}
Bytes bmp(int w,int h){int stride=(w*3+3)&~3;Bytes b(54+stride*h);b[0]='B';b[1]='M';put(b,2,int(b.size()));put(b,10,54);put(b,14,40);put(b,18,w);put(b,22,-h);b[26]=1;b[28]=24;return b;}
uint8_t* pixel(Bytes& b,int w,int x,int y){return b.data()+54+y*((w*3+3)&~3)+x*3;}
bool visible(const uint8_t* p){return p[0]!=255||p[1]!=0||p[2]!=255;}
uint32_t state=6755;uint8_t random_byte(){state=state*1664525u+1013904223u;return uint8_t(state>>24);}
void paste(Bytes& b,int w,const Bytes& t,int tw,int th,int x,int y){for(int v=0;v<th;++v)for(int u=0;u<tw;++u){const auto* p=t.data()+54+v*((tw*3+3)&~3)+u*3;if(visible(p))memcpy(pixel(b,w,x+u,y+v),p,3);}}

// Deliberately different evaluation from the product's integer-moment formula:
// collect samples, explicitly subtract channel means, then accumulate doubles.
float reference(const Bytes& b,int w,const Bytes& t,int tw,int th,int x,int y){
    std::vector<double> a[3],v[3];bool identical=true;
    for(int ty=0;ty<th;++ty)for(int tx=0;tx<tw;++tx){const auto* tp=t.data()+54+ty*((tw*3+3)&~3)+tx*3;const auto* ip=b.data()+54+(y+ty)*((w*3+3)&~3)+(x+tx)*3;if(!visible(tp))continue;
        for(int c=0;c<3;++c){a[c].push_back(ip[c]);v[c].push_back(tp[c]);identical&=ip[c]==tp[c];}}
    if(a[0].empty())return -1;if(identical)return 1;
    double n=0,ei=0,et=0;
    for(int c=0;c<3;++c){const double ma=std::accumulate(a[c].begin(),a[c].end(),0.0)/a[c].size(),mt=std::accumulate(v[c].begin(),v[c].end(),0.0)/v[c].size();
        for(size_t k=0;k<a[c].size();++k){double da=a[c][k]-ma,dt=v[c][k]-mt;n+=da*dt;ei+=da*da;et+=dt*dt;}}
    if(ei<=1e-12||et<=1e-12)return -1;const double r=n/std::sqrt(ei*et);return r>1-1e-6?1:float(std::clamp(r,-1.,1.));
}
struct Hit{int x,y;float score;};
bool order(Hit a,Hit b){return a.score!=b.score?a.score>b.score:a.y!=b.y?a.y<b.y:a.x<b.x;}
float overlap(Hit a,Hit b,int w,int h){int iw=std::max(0,std::min(a.x+w,b.x+w)-std::max(a.x,b.x)),ih=std::max(0,std::min(a.y+h,b.y+h)-std::max(a.y,b.y));return float(double(iw*ih)/(2*w*h-iw*ih));}

int main(){
    constexpr int w=53,h=39,tw=17,th=13;
    const auto dir=std::filesystem::temp_directory_path()/("cv-mask-oracle-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directories(dir);
    struct Guard{std::filesystem::path p;~Guard(){std::error_code e;std::filesystem::remove_all(p,e);}} guard{dir};
    int handle=0;assert(CV_Create(&handle)==0);
    for(int seed=0;seed<8;++seed){auto t=bmp(tw,th),b=bmp(w,h);
        for(size_t k=54;k<b.size();++k)b[k]=random_byte();
        for(int y=0;y<th;++y)for(int x=0;x<tw;++x){auto* p=pixel(t,tw,x,y);p[0]=255;p[1]=0;p[2]=255;if((x*13+y*7+seed)%11==0){p[0]=uint8_t(80+random_byte()%120);p[1]=p[2]=255;}}
        paste(b,w,t,tw,th,21,15);
        const auto path=dir/"sparse.bmp";{std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(t.data()),t.size());}
        assert(CV_LoadTemplateDir(handle,dir.string().c_str(),0)==1);
        for(int variant=0;variant<2;++variant){
            if(variant)for(int y=0;y<th;++y)for(int x=0;x<tw;++x)if(!visible(pixel(t,tw,x,y)))for(int c=0;c<3;++c)pixel(b,w,21+x,15+y)[c]=random_byte();
            for(float threshold:{0.f,.3f,.9f,.99f,1.f}){
                std::vector<Hit> expected;
                for(int y=0;y<=h-th;++y)for(int x=0;x<=w-tw;++x){const float s=reference(b,w,t,tw,th,x,y);if(s>=threshold)expected.push_back({x,y,s});}
                std::sort(expected.begin(),expected.end(),order);assert(!expected.empty());
                CVMatchResult r{};assert(CV_FindTransparentOne(handle,"sparse.bmp",b.data(),int(b.size()),threshold,0,"FF00FF",&r,7,11)==1);
                assert(r.x==expected[0].x+7&&r.y==expected[0].y+11&&r.w==tw&&r.h==th&&std::abs(r.sim-expected[0].score)<1e-6);
                std::vector<Hit> selected;
                for(auto hit:expected)if(std::none_of(selected.begin(),selected.end(),[&](Hit s){return overlap(hit,s,tw,th)>.3f;}))selected.push_back(hit);
                std::sort(selected.begin(),selected.end(),[](Hit a,Hit b){return a.y!=b.y?a.y<b.y:a.x<b.x;});
                std::ostringstream text;for(size_t k=0;k<selected.size();++k){if(k)text<<'|';text<<"0,"<<selected[k].x+7<<','<<selected[k].y+11;}
                const char* actual=CV_FindTransparentMultiText(handle,"sparse.bmp",b.data(),int(b.size()),"",threshold,"FF00FF",7,11);assert(actual&&text.str()==actual&&AI_GetLastError()[0]==0);
#if defined(AIENGINE_CV_TEST_HOOKS)
                CVTest_ForceScalarMasked(1);CVMatchResult scalar{};assert(CV_FindTransparentOne(handle,"sparse.bmp",b.data(),int(b.size()),threshold,0,"FF00FF",&scalar,7,11)==1);assert(memcmp(&scalar,&r,sizeof(r))==0);CVTest_ForceScalarMasked(0);
#endif
            }
        }
    }
    // Tile boundaries, final legal coordinate, same-score ordering and both BMP
    // row directions. Grayscale pixels make both legacy grayscale conversions
    // exact here without importing the product's conversion helper.
    constexpr int large_w=1100,large_h=600;
    auto t=bmp(tw,th),large=bmp(large_w,large_h);
    for(size_t k=54;k<large.size();++k)large[k]=random_byte();
    for(int y=0;y<th;++y)for(int x=0;x<tw;++x){auto* p=pixel(t,tw,x,y);p[0]=255;p[1]=0;p[2]=255;
        if((x*13+y*7)%7==0)p[0]=p[1]=p[2]=random_byte();}
    const auto path=dir/"sparse.bmp";{std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(t.data()),t.size());}
    assert(CV_LoadTemplateDir(handle,dir.string().c_str(),0)==1);
    for(auto xy:{std::pair<int,int>{0,0},{510,255},{large_w-tw,large_h-th}})paste(large,large_w,t,tw,th,xy.first,xy.second);
    for(int direction=0;direction<2;++direction){
        if(direction){const int stride=(large_w*3+3)&~3;put(large,22,large_h);
            for(int y=0;y<large_h/2;++y)for(int x=0;x<stride;++x)std::swap(large[54+y*stride+x],large[54+(large_h-1-y)*stride+x]);}
        for(int mode:{0,1})for(float threshold:{.9f,.99f,std::nextafter(1.f,0.f),1.f}){
            CVMatchResult r{};assert(CV_FindTransparentOne(handle,"sparse.bmp",large.data(),int(large.size()),threshold,mode,"FF00FF",&r,0,0)==1);
            assert(r.x==0&&r.y==0&&r.sim==1.f&&r.w==tw&&r.h==th);
        }
        const char* actual=CV_FindTransparentMultiText(handle,"sparse.bmp",large.data(),int(large.size()),"",1.f,"FF00FF",0,0);
        assert(actual&&std::string(actual)=="0,0,0|0,510,255|0,1083,587");
    }
#if defined(AIENGINE_CV_TEST_HOOKS)
    for(int kind:{1,2,3,4}){
        const auto before=CVTest_Stats();CVTest_Fault(2,kind);CVMatchResult r{};
        assert(CV_FindTransparentOne(handle,"sparse.bmp",large.data(),int(large.size()),.99f,0,"FF00FF",&r,0,0)<0);
        assert(AI_GetLastError()[0]!=0);const auto after=CVTest_Stats();
        assert(after.active==0&&after.discarded==before.discarded+1&&after.idle_bytes<=after.budget);
        assert(CV_FindTransparentOne(handle,"sparse.bmp",large.data(),int(large.size()),.99f,0,"FF00FF",&r,0,0)==1);
        assert(r.x==0&&r.y==0&&r.sim==1&&AI_GetLastError()[0]==0);
    }
#endif
    assert(CV_ClearTemplateCache(handle)==0);assert(CV_Release(handle)==0);
#if defined(AIENGINE_CV_TEST_HOOKS)
    const auto stats=CVTest_Stats();assert(stats.active==0&&stats.live_templates==0&&stats.idle_bytes<=stats.budget);
#endif
}
