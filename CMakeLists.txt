cmake_minimum_required(VERSION 3.20)

project(ai_engine_dll VERSION 0.14.5 LANGUAGES CXX)

option(AIENGINE_BUILD_TESTS "构建冒烟和运行探测测试程序" ON)
option(AIENGINE_WITH_OPENCV "使用静态 OpenCV 5.0 core/imgproc 加速 CV 模板匹配" OFF)
set(AIENGINE_OPENCV_DIR "" CACHE PATH "OpenCV 5.0 静态安装目录下的 OpenCVConfig.cmake 所在目录")
option(AIENGINE_WITH_ONNXRUNTIME "链接 ONNX Runtime，用于真实 YOLO/OCR 后端" OFF)
option(AIENGINE_BUILD_X64_DLL "构建供 x64 宿主直接加载的业务 DLL" OFF)
option(AIENGINE_BUILD_WORKER "构建供 x86 易语言代理使用的 x64 worker" ON)
option(AIENGINE_EMBED_ASSETS "把默认 YOLO/OCR 模型、标签和字典打进目标二进制" OFF)
option(AIENGINE_EMBED_OCR_ASSETS "只把默认 OCR 检测、识别模型和字典打进目标二进制" OFF)
set(AIENGINE_ONNXRUNTIME_DIR "" CACHE PATH "ORT DirectML SDK 根目录，例如 third_party/runtime/ort-directml-1.24.4")
set(AIENGINE_EMBED_YOLO_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/yolo/smc.onnx" CACHE FILEPATH "内置 YOLO ONNX 模型")
set(AIENGINE_EMBED_YOLO_LABELS "${CMAKE_CURRENT_SOURCE_DIR}/models/yolo/smc.txt" CACHE FILEPATH "内置 YOLO 标签文件")
set(AIENGINE_EMBED_OCR_DET_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/ocr_ppocrv6/det.fp32.onnx" CACHE FILEPATH "内置 OCR 检测 ONNX 模型")
set(AIENGINE_EMBED_OCR_REC_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/ocr_ppocrv6/rec.fp32.onnx" CACHE FILEPATH "内置 OCR 识别 ONNX 模型")
set(AIENGINE_EMBED_OCR_CHARSET "${CMAKE_CURRENT_SOURCE_DIR}/models/ocr_ppocrv6/ppocrv6_tiny_dict.txt" CACHE FILEPATH "内置 OCR 字符集文件")

set(AIENGINE_BUILD_API_DLL ON)
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND AIENGINE_WITH_ONNXRUNTIME AND NOT AIENGINE_BUILD_X64_DLL)
    set(AIENGINE_BUILD_API_DLL OFF)
endif()

if(AIENGINE_BUILD_API_DLL)
    add_library(ai_engine SHARED
        src/ai_engine.cpp
        src/backends.cpp
        src/config.cpp
        src/cv_fast.cpp
        src/embedded_assets.cpp
        src/engine.cpp
        src/error.cpp
        src/ocr_reading_order.cpp
        src/ocr_color_filter.cpp
        src/runtime_status.cpp
    )
endif()

if(AIENGINE_BUILD_API_DLL)
    target_compile_features(ai_engine PRIVATE cxx_std_17)
    target_include_directories(ai_engine PUBLIC include)
    target_include_directories(ai_engine PRIVATE src)
endif()

if(AIENGINE_WITH_OPENCV AND AIENGINE_BUILD_API_DLL)
    if(AIENGINE_OPENCV_DIR)
        set(OpenCV_DIR "${AIENGINE_OPENCV_DIR}" CACHE PATH "OpenCV package directory" FORCE)
    endif()
    set(OpenCV_STATIC ON)
    find_package(OpenCV 5.0.0 EXACT REQUIRED COMPONENTS core imgproc)
    target_compile_definitions(ai_engine PRIVATE AIENGINE_WITH_OPENCV=1 AIENGINE_OPENCV_VERSION="${OpenCV_VERSION}")
    target_sources(ai_engine PRIVATE src/memory_zip.cpp)
    target_link_libraries(ai_engine PRIVATE ${OpenCV_LIBS} zlib)
endif()

if(AIENGINE_WITH_ONNXRUNTIME)
    if(NOT AIENGINE_ONNXRUNTIME_DIR)
        message(FATAL_ERROR "AIENGINE_WITH_ONNXRUNTIME=ON 时必须设置 AIENGINE_ONNXRUNTIME_DIR")
    endif()

    if(NOT EXISTS "${AIENGINE_ONNXRUNTIME_DIR}/VERSION_NUMBER")
        message(FATAL_ERROR "AIENGINE_ONNXRUNTIME_DIR is missing VERSION_NUMBER")
    endif()
    file(STRINGS "${AIENGINE_ONNXRUNTIME_DIR}/VERSION_NUMBER" AIENGINE_ORT_VERSION LIMIT_COUNT 1)
    string(STRIP "${AIENGINE_ORT_VERSION}" AIENGINE_ORT_VERSION)
    if(NOT AIENGINE_ORT_VERSION STREQUAL "1.24.4")
        message(FATAL_ERROR "CQ_AI 0.14.5 requires Microsoft.ML.OnnxRuntime.DirectML 1.24.4, got ${AIENGINE_ORT_VERSION}")
    endif()

    if(AIENGINE_BUILD_API_DLL)
        target_sources(ai_engine PRIVATE src/backend_onnxruntime.cpp)
        target_compile_definitions(ai_engine PRIVATE
            AIENGINE_WITH_ONNXRUNTIME=1
            AIENGINE_RUNTIME_FLAVOR_NAME="core")
        target_include_directories(ai_engine PRIVATE "${AIENGINE_ONNXRUNTIME_DIR}/include")

        if(WIN32)
            target_link_libraries(ai_engine PRIVATE
                "${AIENGINE_ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
                d3d12 dxgi gdi32)
            add_custom_command(TARGET ai_engine POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${AIENGINE_ONNXRUNTIME_DIR}/bin"
                    "$<TARGET_FILE_DIR:ai_engine>")
        else()
            target_link_libraries(ai_engine PRIVATE "${AIENGINE_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
        endif()
    endif()
endif()

if(AIENGINE_EMBED_ASSETS OR AIENGINE_EMBED_OCR_ASSETS)
    if(NOT WIN32)
        message(FATAL_ERROR "AIENGINE_EMBED_ASSETS 当前只支持 Windows 资源打包")
    endif()
    if(NOT AIENGINE_WITH_ONNXRUNTIME AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "AIENGINE_EMBED_ASSETS 需要同时启用 AIENGINE_WITH_ONNXRUNTIME")
    endif()

    set(AIENGINE_REQUIRED_EMBED_ASSETS
        "${AIENGINE_EMBED_OCR_DET_MODEL}"
        "${AIENGINE_EMBED_OCR_REC_MODEL}"
        "${AIENGINE_EMBED_OCR_CHARSET}")
    if(AIENGINE_EMBED_ASSETS)
        list(APPEND AIENGINE_REQUIRED_EMBED_ASSETS
            "${AIENGINE_EMBED_YOLO_MODEL}"
            "${AIENGINE_EMBED_YOLO_LABELS}")
    endif()

    foreach(asset_file IN LISTS AIENGINE_REQUIRED_EMBED_ASSETS)
        if(NOT EXISTS "${asset_file}")
            message(FATAL_ERROR "内置资源文件不存在：${asset_file}")
        endif()
    endforeach()

    file(TO_CMAKE_PATH "${AIENGINE_EMBED_OCR_REC_MODEL}" AIENGINE_EMBED_OCR_REC_MODEL_RC)
    file(TO_CMAKE_PATH "${AIENGINE_EMBED_OCR_CHARSET}" AIENGINE_EMBED_OCR_CHARSET_RC)
    file(TO_CMAKE_PATH "${AIENGINE_EMBED_OCR_DET_MODEL}" AIENGINE_EMBED_OCR_DET_MODEL_RC)
    if(AIENGINE_EMBED_ASSETS)
        file(TO_CMAKE_PATH "${AIENGINE_EMBED_YOLO_MODEL}" AIENGINE_EMBED_YOLO_MODEL_RC)
        file(TO_CMAKE_PATH "${AIENGINE_EMBED_YOLO_LABELS}" AIENGINE_EMBED_YOLO_LABELS_RC)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_assets.rc.in"
            "${CMAKE_CURRENT_BINARY_DIR}/embedded_assets.rc"
            @ONLY)
    else()
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_ocr_assets.rc.in"
            "${CMAKE_CURRENT_BINARY_DIR}/embedded_assets.rc"
            @ONLY)
    endif()
    set(AIENGINE_EMBEDDED_RESOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/embedded_assets.rc")
    if(AIENGINE_BUILD_API_DLL)
        target_sources(ai_engine PRIVATE "${AIENGINE_EMBEDDED_RESOURCE_FILE}")
        target_compile_definitions(ai_engine PRIVATE AIENGINE_EMBED_ASSETS=1)
    endif()
endif()

if(AIENGINE_BUILD_API_DLL)
    if(MSVC)
        target_compile_options(ai_engine PRIVATE /O2 /GL /EHsc /permissive- /W4 /utf-8)
        target_link_options(ai_engine PRIVATE /LTCG)
        set_property(TARGET ai_engine PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        target_compile_options(ai_engine PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
endif()

if(AIENGINE_BUILD_API_DLL)
    set(AIENGINE_API_OUTPUT_NAME "CQ_X86")
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(AIENGINE_API_OUTPUT_NAME "CQ_AI_x64")
        target_compile_definitions(ai_engine PRIVATE AIENGINE_BINARY_NAME="CQ_AI_x64")
    endif()
    set_target_properties(ai_engine PROPERTIES
        OUTPUT_NAME "${AIENGINE_API_OUTPUT_NAME}"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

if(AIENGINE_BUILD_TESTS AND AIENGINE_BUILD_API_DLL)
    enable_testing()

    function(ai_engine_configure_test target)
        target_compile_features(${target} PRIVATE cxx_std_17)
        target_include_directories(${target} PRIVATE include)
        target_link_libraries(${target} PRIVATE ai_engine)
        if(MSVC)
            target_compile_options(${target} PRIVATE /EHsc /utf-8 /UNDEBUG)
        endif()
    endfunction()

    add_executable(ai_engine_smoke tests/smoke_test.cpp)
    ai_engine_configure_test(ai_engine_smoke)
    add_test(NAME ai_engine_smoke COMMAND ai_engine_smoke)
    set_tests_properties(ai_engine_smoke PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(ai_engine_e_api_smoke tests/e_api_smoke.cpp)
    ai_engine_configure_test(ai_engine_e_api_smoke)
    add_test(NAME ai_engine_e_api_smoke COMMAND ai_engine_e_api_smoke)
    set_tests_properties(ai_engine_e_api_smoke PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(ai_engine_ocr_color_filter_test
        tests/ocr_color_filter_test.cpp
        src/ocr_color_filter.cpp)
    target_compile_features(ai_engine_ocr_color_filter_test PRIVATE cxx_std_17)
    target_include_directories(ai_engine_ocr_color_filter_test PRIVATE include src)
    if(MSVC)
        target_compile_options(ai_engine_ocr_color_filter_test PRIVATE /EHsc /utf-8 /UNDEBUG)
    endif()
    add_test(NAME ai_engine_ocr_color_filter_test COMMAND ai_engine_ocr_color_filter_test)

    add_executable(ai_engine_ocr_reading_order_test
        tests/ocr_reading_order_test.cpp
        src/ocr_reading_order.cpp)
    target_compile_features(ai_engine_ocr_reading_order_test PRIVATE cxx_std_17)
    target_include_directories(ai_engine_ocr_reading_order_test PRIVATE src)
    if(MSVC)
        target_compile_options(ai_engine_ocr_reading_order_test PRIVATE /EHsc /utf-8 /UNDEBUG)
    endif()
    add_test(NAME ai_engine_ocr_reading_order_test COMMAND ai_engine_ocr_reading_order_test)

    if(WIN32 AND (AIENGINE_EMBED_OCR_ASSETS OR AIENGINE_EMBED_ASSETS OR CMAKE_SIZEOF_VOID_P EQUAL 4))
        add_executable(ai_engine_ocr_business_test tests/ocr_business_test.cpp)
        ai_engine_configure_test(ai_engine_ocr_business_test)
        add_test(NAME ai_engine_ocr_business_test COMMAND ai_engine_ocr_business_test)
        set_tests_properties(ai_engine_ocr_business_test PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 300 RUN_SERIAL TRUE)

        add_executable(ai_engine_ocr_reading_order_business_test tests/ocr_reading_order_business_test.cpp)
        ai_engine_configure_test(ai_engine_ocr_reading_order_business_test)
        add_test(NAME ai_engine_ocr_reading_order_business_test COMMAND ai_engine_ocr_reading_order_business_test)
        set_tests_properties(ai_engine_ocr_reading_order_business_test PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 600 RUN_SERIAL TRUE)
    endif()

    if(AIENGINE_WITH_OPENCV)
        add_executable(ai_engine_compact_result_test
            tests/compact_result_test.cpp)
        ai_engine_configure_test(ai_engine_compact_result_test)
        target_include_directories(ai_engine_compact_result_test PRIVATE src)
        add_test(NAME ai_engine_compact_result_test
            COMMAND ai_engine_compact_result_test)

        add_executable(ai_engine_origin_zip_test tests/origin_zip_test.cpp)
        ai_engine_configure_test(ai_engine_origin_zip_test)
        target_include_directories(ai_engine_origin_zip_test PRIVATE src)
        add_test(NAME ai_engine_origin_zip_test COMMAND ai_engine_origin_zip_test)
        set_tests_properties(ai_engine_origin_zip_test PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 120 RUN_SERIAL TRUE)

        add_executable(ai_engine_cv_business_benchmark tests/cv_business_benchmark.cpp)
        ai_engine_configure_test(ai_engine_cv_business_benchmark)
        add_executable(ai_engine_acp_compat_test tests/acp_compat_test.cpp)
        ai_engine_configure_test(ai_engine_acp_compat_test)
        add_test(NAME ai_engine_acp_compat_test
            COMMAND ai_engine_acp_compat_test)
        add_test(NAME ai_engine_cv_single_benchmark
            COMMAND ai_engine_cv_business_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cv/base" 1000 --enforce single)
        add_test(NAME ai_engine_cv_multi_text_benchmark
            COMMAND ai_engine_cv_business_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cv/base" 1000 --enforce multi_text)
        set_tests_properties(ai_engine_compact_result_test ai_engine_acp_compat_test ai_engine_cv_single_benchmark ai_engine_cv_multi_text_benchmark PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 120 RUN_SERIAL TRUE)
    endif()

    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
        add_executable(ai_engine_proxy_worker_smoke tests/proxy_worker_smoke.cpp)
        ai_engine_configure_test(ai_engine_proxy_worker_smoke)
        target_compile_definitions(ai_engine_proxy_worker_smoke PRIVATE AIENGINE_TEST_EMBED_OCR=1)
        add_test(NAME ai_engine_proxy_worker_smoke COMMAND ai_engine_proxy_worker_smoke)
        set_tests_properties(ai_engine_proxy_worker_smoke PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 300)

        add_executable(ai_engine_ocr_small_multiline_regression tests/ocr_small_multiline_regression.cpp)
        ai_engine_configure_test(ai_engine_ocr_small_multiline_regression)
        add_test(NAME ai_engine_ocr_small_multiline_regression COMMAND ai_engine_ocr_small_multiline_regression)
        set_tests_properties(ai_engine_ocr_small_multiline_regression PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 300 RUN_SERIAL TRUE)

        add_executable(ai_engine_ocr_robustness_test
            tests/ocr_robustness_test.cpp)
        ai_engine_configure_test(ai_engine_ocr_robustness_test)
        set(AIENGINE_OCR_ROBUSTNESS_FIXTURE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ocr/antialias_multiline.bmp")
        if(EXISTS "${AIENGINE_OCR_ROBUSTNESS_FIXTURE}")
            add_test(NAME ai_engine_ocr_robustness_cpu
                COMMAND ai_engine_ocr_robustness_test
                    "${AIENGINE_OCR_ROBUSTNESS_FIXTURE}" 2)
            add_test(NAME ai_engine_ocr_robustness_dml
                COMMAND ai_engine_ocr_robustness_test
                    "${AIENGINE_OCR_ROBUSTNESS_FIXTURE}" 1)
            add_test(NAME ai_engine_ocr_robustness_auto
                COMMAND ai_engine_ocr_robustness_test
                    "${AIENGINE_OCR_ROBUSTNESS_FIXTURE}" 0)
            set_tests_properties(
                ai_engine_ocr_robustness_cpu
                ai_engine_ocr_robustness_dml
                ai_engine_ocr_robustness_auto
                PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    TIMEOUT 900
                    RUN_SERIAL TRUE)
        endif()

    endif()

    if(AIENGINE_WITH_ONNXRUNTIME AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        add_executable(ai_engine_runtime_probe tests/runtime_probe.cpp)
        ai_engine_configure_test(ai_engine_runtime_probe)
        add_test(NAME ai_engine_runtime_probe COMMAND ai_engine_runtime_probe)
        set_tests_properties(ai_engine_runtime_probe PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 180)

        add_executable(ai_engine_embedded_benchmark tests/embedded_benchmark.cpp)
        ai_engine_configure_test(ai_engine_embedded_benchmark)
        add_test(NAME ai_engine_embedded_benchmark COMMAND ai_engine_embedded_benchmark 1)
        set_tests_properties(ai_engine_embedded_benchmark PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 180)
    endif()
endif()

if(WIN32 AND AIENGINE_WITH_ONNXRUNTIME AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND AIENGINE_BUILD_WORKER)
    if(NOT AIENGINE_EMBED_ASSETS AND NOT AIENGINE_EMBED_OCR_ASSETS)
        message(FATAL_ERROR "CQ_AI_worker 必须启用 AIENGINE_EMBED_OCR_ASSETS 或 AIENGINE_EMBED_ASSETS")
    endif()
    set(AIENGINE_RUNTIME_BIN_DIR "${AIENGINE_ONNXRUNTIME_DIR}/bin")
    set(AIENGINE_RUNTIME_LICENSE_DIR "${AIENGINE_ONNXRUNTIME_DIR}/licenses")
    set(AIENGINE_RUNTIME_FILES
        "${AIENGINE_RUNTIME_BIN_DIR}/onnxruntime.dll"
        "${AIENGINE_RUNTIME_BIN_DIR}/DirectML.dll"
        "${AIENGINE_RUNTIME_BIN_DIR}/msvcp140.dll"
        "${AIENGINE_RUNTIME_BIN_DIR}/msvcp140_1.dll"
        "${AIENGINE_RUNTIME_BIN_DIR}/vcruntime140.dll"
        "${AIENGINE_RUNTIME_BIN_DIR}/vcruntime140_1.dll")
    set(AIENGINE_NOTICE_FILES
        "${AIENGINE_RUNTIME_LICENSE_DIR}/onnxruntime-LICENSE.txt"
        "${AIENGINE_RUNTIME_LICENSE_DIR}/onnxruntime-ThirdPartyNotices.txt"
        "${AIENGINE_RUNTIME_LICENSE_DIR}/directml-LICENSE.txt"
        "${AIENGINE_RUNTIME_LICENSE_DIR}/directml-LICENSE-CODE.txt"
        "${AIENGINE_RUNTIME_LICENSE_DIR}/directml-ThirdPartyNotices.txt"
        "${AIENGINE_RUNTIME_LICENSE_DIR}/visual-cpp-runtime-license.txt")
    foreach(runtime_file IN LISTS AIENGINE_RUNTIME_FILES AIENGINE_NOTICE_FILES)
        if(NOT EXISTS "${runtime_file}")
            message(FATAL_ERROR
                "Runtime staging is incomplete: ${runtime_file}\n"
                "Run scripts/prepare_dependencies.ps1 first.")
        endif()
    endforeach()

    add_executable(cq_runtime_packer
        tools/runtime_packer.cpp)
    target_compile_features(cq_runtime_packer PRIVATE cxx_std_17)
    target_include_directories(cq_runtime_packer PRIVATE src)
    target_link_libraries(cq_runtime_packer PRIVATE cabinet bcrypt)
    if(MSVC)
        target_compile_options(cq_runtime_packer PRIVATE /O2 /EHsc /permissive- /W4 /utf-8)
        set_property(TARGET cq_runtime_packer PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()

    set(AIENGINE_RUNTIME_BUNDLE_FILE "${CMAKE_CURRENT_BINARY_DIR}/cq_runtime.bin")
    add_custom_command(
        OUTPUT "${AIENGINE_RUNTIME_BUNDLE_FILE}"
        COMMAND $<TARGET_FILE:cq_runtime_packer>
            "${AIENGINE_RUNTIME_BUNDLE_FILE}"
            --runtime "onnxruntime.dll=${AIENGINE_RUNTIME_BIN_DIR}/onnxruntime.dll"
            --directml "DirectML.dll=${AIENGINE_RUNTIME_BIN_DIR}/DirectML.dll"
            --runtime "msvcp140.dll=${AIENGINE_RUNTIME_BIN_DIR}/msvcp140.dll"
            --runtime "msvcp140_1.dll=${AIENGINE_RUNTIME_BIN_DIR}/msvcp140_1.dll"
            --runtime "vcruntime140.dll=${AIENGINE_RUNTIME_BIN_DIR}/vcruntime140.dll"
            --runtime "vcruntime140_1.dll=${AIENGINE_RUNTIME_BIN_DIR}/vcruntime140_1.dll"
            --notice "onnxruntime-LICENSE.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/onnxruntime-LICENSE.txt"
            --notice "onnxruntime-ThirdPartyNotices.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/onnxruntime-ThirdPartyNotices.txt"
            --notice "directml-LICENSE.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/directml-LICENSE.txt"
            --notice "directml-LICENSE-CODE.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/directml-LICENSE-CODE.txt"
            --notice "directml-ThirdPartyNotices.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/directml-ThirdPartyNotices.txt"
            --notice "visual-cpp-runtime-license.txt=${AIENGINE_RUNTIME_LICENSE_DIR}/visual-cpp-runtime-license.txt"
        DEPENDS cq_runtime_packer ${AIENGINE_RUNTIME_FILES} ${AIENGINE_NOTICE_FILES}
        VERBATIM)
    add_custom_target(cq_runtime_bundle DEPENDS "${AIENGINE_RUNTIME_BUNDLE_FILE}")
    file(TO_CMAKE_PATH "${AIENGINE_RUNTIME_BUNDLE_FILE}" AIENGINE_RUNTIME_BUNDLE_FILE_RC)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/embedded_runtime.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/embedded_runtime.rc"
        @ONLY)
    set(AIENGINE_EMBEDDED_RUNTIME_RESOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/embedded_runtime.rc")
    set_property(SOURCE "${AIENGINE_EMBEDDED_RUNTIME_RESOURCE_FILE}" APPEND PROPERTY
        OBJECT_DEPENDS "${AIENGINE_RUNTIME_BUNDLE_FILE}")

    add_executable(cq_ai_worker
        src/worker_main.cpp
        src/backends.cpp
        src/backend_onnxruntime.cpp
        src/config.cpp
        src/embedded_assets.cpp
        src/engine.cpp
        src/error.cpp
        src/ocr_reading_order.cpp
        src/ocr_color_filter.cpp
        src/runtime_bundle_loader.cpp
        src/runtime_status.cpp
        ${AIENGINE_EMBEDDED_RESOURCE_FILE}
        ${AIENGINE_EMBEDDED_RUNTIME_RESOURCE_FILE}
    )
    add_dependencies(cq_ai_worker cq_runtime_bundle)
    target_compile_features(cq_ai_worker PRIVATE cxx_std_17)
    target_compile_definitions(cq_ai_worker PRIVATE
        AIENGINE_WITH_ONNXRUNTIME=1
        AIENGINE_EMBED_ASSETS=1
        AIENGINE_EMBEDDED_ORT_RUNTIME=1
        AIENGINE_RUNTIME_FLAVOR_NAME="core"
        ORT_API_MANUAL_INIT=1)
    target_include_directories(cq_ai_worker PRIVATE include src "${AIENGINE_ONNXRUNTIME_DIR}/include")
    target_link_libraries(cq_ai_worker PRIVATE cabinet bcrypt d3d12 dxgi gdi32)
    if(MSVC)
        target_compile_options(cq_ai_worker PRIVATE /O2 /GL /EHsc /permissive- /W4 /utf-8)
        target_link_options(cq_ai_worker PRIVATE /LTCG)
        set_property(TARGET cq_ai_worker PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    set_target_properties(cq_ai_worker PROPERTIES OUTPUT_NAME "CQ_AI_worker")
endif()
