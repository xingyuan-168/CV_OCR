#include "cv_correlation.h"
#include "cv_masked_score_avx2.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace ai::cv_detail;
using Clock=std::chrono::steady_clock;
cv::Mat bmp(const std::filesystem::path& path){
 std::ifstream f(path,std::ios::binary);std::vector<unsigned char>b{std::istreambuf_iterator<char>(f),{}};
 if(b.size()<54||b[28]!=24)throw std::runtime_error("24bit BMP required");
 int w,h,o;memcpy(&w,b.data()+18,4);memcpy(&h,b.data()+22,4);memcpy(&o,b.data()+10,4);int s=(w*3+3)&~3;
 cv::Mat m(abs(h),w,CV_8UC3);for(int y=0;y<m.rows;++y)memcpy(m.ptr(y),b.data()+o+(h>0?h-1-y:y)*s,w*3);return m;
}
template<class F> double bench(F f){std::vector<double>v;for(int i=0;i<8;++i)f();for(int i=0;i<64;++i){auto b=Clock::now();f();v.push_back(std::chrono::duration<double,std::milli>(Clock::now()-b).count());}std::sort(v.begin(),v.end());return v[60];}
size_t survivors(const cv::Mat&m){size_t n=0;for(int y=0;y<m.rows;++y)for(int x=0;x<m.cols;++x)n+=m.at<float>(y,x)>0;return n;}
struct Audit{size_t zero_energy=0,constant_rectangle=0,qualified=0;};
Audit audit(const cv::Mat&im,const cv::Mat&t,const MaskedTemplateSpectrum&s,const cv::Mat&bounds){Audit a;const int ch=im.channels();uint64_t ts[3]{};for(auto p:s.visible)for(int c=0;c<ch;++c)ts[c]+=t.ptr(p.y)[p.x*ch+c];
 for(int y=0;y<bounds.rows;++y)for(int x=0;x<bounds.cols;++x){if(bounds.at<float>(y,x)<=0)continue;uint64_t sums[3]{},squares[3]{},dots[3]{};bool same=true;
  for(auto p:s.visible)for(int c=0;c<ch;++c){uint32_t v=im.ptr(y+p.y)[(x+p.x)*ch+c],tv=t.ptr(p.y)[p.x*ch+c];sums[c]+=v;squares[c]+=v*v;dots[c]+=v*tv;same&=v==tv;}
  double e=0,n=0;for(int c=0;c<ch;++c){e+=double(squares[c])-double(sums[c])*sums[c]/s.count;n+=double(dots[c])-double(sums[c])*ts[c]/s.count;}
  if(e<=1e-12){++a.zero_energy;bool flat=true;for(int j=0;j<t.rows&&flat;++j)for(int i=0;i<t.cols&&flat;++i)for(int c=0;c<ch;++c)if(im.ptr(y+j)[(x+i)*ch+c]!=im.ptr(y)[x*ch+c]){flat=false;break;}a.constant_rectangle+=flat;}
  a.qualified+=same||(e>1e-12&&s.energy>1e-12&&n/std::sqrt(e*s.energy)>=double(.9f));
 }return a;}
void run(const char*name,cv::Mat image,cv::Mat templ,int mode,bool randomize){
 cv::Mat mask(templ.size(),CV_8U);for(int y=0;y<templ.rows;++y)for(int x=0;x<templ.cols;++x){auto p=templ.at<cv::Vec3b>(y,x);mask.at<unsigned char>(y,x)=p!=cv::Vec3b(255,0,255)?255:0;}
 if(randomize){uint32_t s=0x6755;for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols*3;++x){s=1664525u*s+1013904223u;image.ptr(y)[x]=uint8_t(s>>24);}}
 if(mode){cv::Mat t(templ.size(),CV_8U);for(int y=0;y<templ.rows;++y)for(int x=0;x<templ.cols;++x){auto p=templ.at<cv::Vec3b>(y,x);t.at<unsigned char>(y,x)=(29*p[0]+150*p[1]+77*p[2])>>8;}templ=t;cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);}
 auto spec=make_masked_spectrum(templ,mask,spectrum_geometry(image.size()));MaskedCorrelationWorkspace ws;cv::Mat bounds;
 const double prep=bench([&]{ws.invalidate();ws.prepare(image,spec);});
 const double numerator=bench([&]{ws.upper_bounds(image,spec,0.f,bounds);});
 const double warm=bench([&]{ws.upper_bounds(image,spec,.9f,bounds);});size_t full_count=survivors(bounds);
 const auto detail=audit(image,templ,spec,bounds);
 std::vector<uint32_t> offsets;for(auto p:spec.pairs){offsets.push_back(uint32_t(p.first.y*image.step+p.first.x*image.channels()));offsets.push_back(uint32_t(p.second.y*image.step+p.second.x*image.channels()));}
 const double subset=bench([&]{for(int y=0;y<bounds.rows;++y)masked_pair_candidates_avx2(image.ptr(y),image.channels(),(1u<<image.channels())-1,offsets.data(),spec.pairs.size(),ws.numerator.ptr<float>(y),spec.numerator_error,std::pow(double(.9f)-1e-6,2)*spec.energy,bounds.cols,bounds.ptr<float>(y));});
 std::cout<<std::setprecision(9)<<"{\"name\":\""<<name<<"\",\"arch\":\""<<(sizeof(void*)==4?"x86":"x64")<<"\",\"channels\":"<<image.channels()<<",\"width\":"<<image.cols<<",\"height\":"<<image.rows<<",\"geometry\":["<<spec.geometry.width<<','<<spec.geometry.height<<"],\"visible\":"<<spec.count<<",\"energy\":"<<spec.energy<<",\"error_guard\":"<<spec.numerator_error<<",\"positions\":"<<bounds.total()<<",\"survivors32\":"<<full_count<<",\"prepare_p95_ms\":"<<prep<<",\"numerator_p95_ms\":"<<numerator<<",\"subset_only_p95_ms\":"<<subset<<",\"warm_bound_p95_ms\":"<<warm<<",\"subset_variants\":[";
 bool first=true;for(size_t points:{size_t(64),size_t(128),spec.count-(spec.count%2)}){auto test=spec;test.pairs.clear();const auto pairs=points/2;for(size_t k=0;k<pairs;++k){size_t a=k*spec.count/(2*pairs);test.pairs.emplace_back(spec.visible[a],spec.visible[spec.count-1-a]);}double ms=bench([&]{ws.upper_bounds(image,test,.9f,bounds);});if(!first)std::cout<<',';first=false;std::cout<<"{\"points\":"<<points<<",\"survivors\":"<<survivors(bounds)<<",\"p95_ms\":"<<ms<<'}';}
 auto unsafe=spec;unsafe.numerator_error=0;ws.upper_bounds(image,unsafe,.9f,bounds);std::cout<<"],\"unsafe_zero_guard_survivors32\":"<<survivors(bounds)<<",\"zero_energy_survivors\":"<<detail.zero_energy<<",\"constant_rectangle_survivors\":"<<detail.constant_rectangle<<",\"exact_qualified_survivors\":"<<detail.qualified<<"}\n"<<std::flush;
}
int main(int argc,char**argv){try{if(argc!=2)return 2;std::filesystem::path p=std::filesystem::u8path(argv[1]);const auto a=bmp(p/"1.bmp"),b=bmp(p/"2.bmp"),old=bmp(p/std::filesystem::u8path(u8"CV_大图.bmp"));
 run("new_color",b,a,0,false);run("new_gray",b,a,1,false);run("new_random_color",b,a,0,true);run("new_random_gray",b,a,1,true);run("old_hit",old,bmp(p/"xyq.bmp"),0,false);run("old_miss",old,bmp(p/"CV_123.bmp"),0,false);
 }catch(const std::exception&e){std::cerr<<e.what();return 2;}return 0;}
