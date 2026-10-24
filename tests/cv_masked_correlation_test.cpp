#include "cv_correlation.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

// Independent exhaustive oracle. No production candidate selection or score
// helper is called. Integer moments avoid reproducing Fourier rounding errors.
double oracle(const cv::Mat& image,const cv::Mat& t,const cv::Mat& mask,int x,int y) {
    uint64_t s[3]{},st[3]{},q[3]{},qt[3]{},dot[3]{};int n=0;bool same=true;
    for(int v=0;v<t.rows;++v) for(int u=0;u<t.cols;++u) if(mask.at<uint8_t>(v,u)) {
        ++n;
        for(int c=0;c<t.channels();++c) {
            const uint32_t a=image.ptr<uint8_t>(y+v)[(x+u)*t.channels()+c];
            const uint32_t b=t.ptr<uint8_t>(v)[u*t.channels()+c];
            same &= a==b;s[c]+=a;st[c]+=b;q[c]+=a*a;qt[c]+=b*b;dot[c]+=a*b;
        }
    }
    if(!n)return -1;if(same)return 1;
    double num=0,ei=0,et=0;
    for(int c=0;c<t.channels();++c){num+=double(dot[c])-double(s[c])*st[c]/n;ei+=double(q[c])-double(s[c])*s[c]/n;et+=double(qt[c])-double(st[c])*st[c]/n;}
    if(ei<=1e-12||et<=1e-12)return -1;
    const double value=num/std::sqrt(ei*et);return value>1-1e-6?1:std::clamp(value,-1.,1.);
}

int main() {
    cv::setNumThreads(1);cv::RNG rng(0x6755);
    for(int channels:{1,3})for(int seed=0;seed<16;++seed) {
        cv::Mat image(39,53,CV_MAKETYPE(CV_8U,channels));rng.fill(image,cv::RNG::UNIFORM,0,256);
        cv::Mat t(13,17,image.type());rng.fill(t,cv::RNG::UNIFORM,0,256);
        cv::Mat mask(t.size(),CV_8U,cv::Scalar(0));
        for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)if((x*13+y*7+seed)%11==0)mask.at<uint8_t>(y,x)=255;
        if(seed==0)t.setTo(cv::Scalar::all(128));
        if(seed==1)for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)for(int c=0;c<channels;++c)t.ptr<uint8_t>(y)[x*channels+c]=uint8_t(128+((x+y+c)%2));
        if(seed==2&&channels==3)for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x){t.ptr<uint8_t>(y)[x*3+1]=255;t.ptr<uint8_t>(y)[x*3+2]=255;}
        const cv::Point hit(21,15);
        for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)if(mask.at<uint8_t>(y,x))
            for(int c=0;c<channels;++c)image.ptr<uint8_t>(hit.y+y)[(hit.x+x)*channels+c]=t.ptr<uint8_t>(y)[x*channels+c];
        auto spectrum=ai::cv_detail::make_masked_spectrum(t,mask,ai::cv_detail::spectrum_geometry(image.size()));
        ai::cv_detail::MaskedCorrelationWorkspace workspace;
        for(float threshold:{0.f,.3f,.9f,.99f,1.f}) {
            cv::Mat bound;workspace.upper_bounds(image,spectrum,threshold,bound);
            for(int y=0;y<bound.rows;++y)for(int x=0;x<bound.cols;++x) {
                const double expected=oracle(image,t,mask,x,y);
                if(expected>=threshold)assert(bound.at<float>(y,x)>=std::max(0.f,threshold-1e-6f));
                // Every finite, non-excluded Fourier/pair estimate must be an
                // upper bound on the unrounded correlation (allow score clamp).
                if(bound.at<float>(y,x)>=0)assert(bound.at<float>(y,x)+1e-6>=expected);
            }
            assert(oracle(image,t,mask,hit.x,hit.y)==1);
            assert(bound.at<float>(hit)>=threshold-1e-6);
        }
        for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)if(!mask.at<uint8_t>(y,x))
            for(int c=0;c<channels;++c)image.ptr<uint8_t>(hit.y+y)[(hit.x+x)*channels+c]=uint8_t(rng.uniform(0,256));
        workspace.invalidate();cv::Mat bound;workspace.upper_bounds(image,spectrum,.99f,bound);
        assert(oracle(image,t,mask,hit.x,hit.y)==1&&bound.at<float>(hit)>=.99f);
        workspace.discard();assert(workspace.retained_bytes()==0);
    }
    // Projection directions must handle local constants, near-constants,
    // positive affine matches and score rounding near the requested threshold.
    for(int channels:{1,3})for(int seed=0;seed<8;++seed){
        cv::Mat t(17,19,CV_MAKETYPE(CV_8U,channels)),mask(t.size(),CV_8U,cv::Scalar(255));
        rng.fill(t,cv::RNG::UNIFORM,0,80);
        if(seed<2){t.setTo(cv::Scalar::all(254));t.ptr<uint8_t>(8)[9*channels]=253;}
        if(seed==2)mask.setTo(0);
        if(seed==3)for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)if((x+y)%3)mask.at<uint8_t>(y,x)=0;
        cv::Mat image(23,29,t.type(),cv::Scalar::all(seed%2?255:0));
        for(int y=0;y<t.rows;++y)for(int x=0;x<t.cols;++x)if(mask.at<uint8_t>(y,x))for(int c=0;c<channels;++c){int v=t.ptr<uint8_t>(y)[x*channels+c];image.ptr<uint8_t>(y+3)[(x+4)*channels+c]=uint8_t(seed>=4?2*v+3:v);}
        auto spectrum=ai::cv_detail::make_masked_spectrum(t,mask,ai::cv_detail::spectrum_geometry(image.size()));
        ai::cv_detail::MaskedCorrelationWorkspace workspace;
        std::vector<float> thresholds={.9f,.99f,1.f,std::nextafter(1.f,0.f)};
        for(int y=0;y<=image.rows-t.rows;++y)for(int x=0;x<=image.cols-t.cols;++x){float v=float(oracle(image,t,mask,x,y));if(v>0&&v<1){thresholds.push_back(v);thresholds.push_back(std::nextafter(v,0.f));thresholds.push_back(std::nextafter(v,1.f));}}
        for(bool optimized:{false,true}){cv::setUseOptimized(optimized);workspace.invalidate();
            for(float threshold:thresholds){cv::Mat bound;workspace.upper_bounds(image,spectrum,threshold,bound);
                for(int y=0;y<bound.rows;++y)for(int x=0;x<bound.cols;++x){float expected=float(oracle(image,t,mask,x,y));if(expected>=threshold)assert(bound.at<float>(y,x)>=0);}}
        }
        // Change content without changing the allocation and explicitly signal
        // invalidation, as the public byte-comparison cache does every frame.
        image.setTo(cv::Scalar::all(73));workspace.invalidate();cv::Mat bound;workspace.upper_bounds(image,spectrum,.9f,bound);
        for(int y=0;y<bound.rows;++y)for(int x=0;x<bound.cols;++x)if(float(oracle(image,t,mask,x,y))>=.9f)assert(bound.at<float>(y,x)>=0);
        workspace.discard();assert(workspace.retained_bytes()==0);
    }
    cv::setUseOptimized(true);
    std::cout<<"Independent exhaustive masked upper-bound and invariance checks passed\n";
}
