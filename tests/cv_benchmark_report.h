#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

// Test-only raw samples. The runner binds the loaded adjacent DLL hash before
// and after each process, and computes nearest-rank statistics without rounding.
inline double cv_benchmark_cold_ms=0;
inline void cv_benchmark_report(const char* name, const std::vector<double>& samples) {
    const char* path=std::getenv("CQ_CV_BENCHMARK_REPORT");
    if (!path || !*path) return;
    std::ofstream file(path,std::ios::app);
    if (!file) throw std::runtime_error("Cannot write benchmark samples");
    file<<std::setprecision(17)<<"{\"name\":\""<<name<<"\",\"prewarm_first_call_ms\":"<<cv_benchmark_cold_ms<<",\"warmup\":64,\"errors\":0,\"samples_ms\":[";
    for(size_t k=0;k<samples.size();++k){if(k)file<<',';file<<samples[k];}
    file<<"]}\n";
}
