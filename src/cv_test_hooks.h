#pragma once
// Private test ABI. Never compiled into candidate/production DLLs.
#include <cstdint>
#if defined(_WIN32)
#define CV_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define CV_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif
struct CVTestStats {
    uint64_t limit, budget, active, peak_active, idle_bytes, discarded, live_templates;
};
CV_TEST_EXPORT void CVTest_Fault(int stage, int kind);
CV_TEST_EXPORT CVTestStats CVTest_Stats();
CV_TEST_EXPORT void CVTest_OverBudget();
CV_TEST_EXPORT void CVTest_Pause(int enabled);
CV_TEST_EXPORT int CVTest_Paused();
CV_TEST_EXPORT void CVTest_LastTimes(double* values);
CV_TEST_EXPORT void CVTest_Reference(int enabled);
