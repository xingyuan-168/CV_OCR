#pragma once

#include <string>

namespace ai_runtime {

// Verifies and extracts the CPU-side ORT runtime, loads it from the private
// per-user cache, and initializes the manually-bound ORT C++ API.
bool initialize_embedded_runtime(std::string* error);

// Extracts DirectML.dll on demand. CPU-only operation never calls this.
bool ensure_embedded_directml(std::string* error);

// Verifies every embedded entry by decompressing it and checking SHA-256.
bool verify_embedded_runtime(std::string* report, std::string* error);

// Returns all embedded third-party notices without writing them to disk.
bool embedded_third_party_notices(std::string* notices, std::string* error);

std::string runtime_cache_path_utf8();
std::string runtime_bundle_hash();
std::string loaded_ort_path_utf8();
bool directml_is_cached();

} // namespace ai_runtime
