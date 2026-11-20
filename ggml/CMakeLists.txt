cmake_minimum_required(VERSION 3.16)
project(ggml-vc4 C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The ggml tree this backend is compiled against (headers + libggml.so).
# Defaults: the vendored llama.cpp from llama-cpp-python 0.3.37 (source
# headers) and the CPU-only build in ~/scratch/build/ggml-shared (library).
set(GGML_ROOT "$ENV{GGML_ROOT}" CACHE PATH "ggml source tree (contains include/ggml.h)")
if(NOT GGML_ROOT)
    set(GGML_ROOT "$ENV{HOME}/scratch/src/llama_cpp_python-0.3.37/vendor/llama.cpp/ggml")
endif()
set(GGML_LIB_DIR "$ENV{GGML_LIB_DIR}" CACHE PATH "directory containing libggml.so")
if(NOT GGML_LIB_DIR)
    set(GGML_LIB_DIR "$ENV{HOME}/scratch/build/ggml-shared/src")
endif()

set(VC4INFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../infer" CACHE PATH "libvc4infer source tree")

find_path(GGML_INCLUDE_DIR ggml.h
          HINTS "${GGML_ROOT}/include" REQUIRED)
find_library(GGML_LIB NAMES ggml
             HINTS "${GGML_LIB_DIR}" "${GGML_LIB_DIR}" REQUIRED)
find_library(GGML_BASE_LIB NAMES ggml-base
             HINTS "${GGML_LIB_DIR}" "${GGML_LIB_DIR}" REQUIRED)
find_library(GGML_CPU_LIB NAMES ggml-cpu
             HINTS "${GGML_LIB_DIR}" "${GGML_LIB_DIR}" REQUIRED)

find_path(VC4INFER_INCLUDE_DIR vc4infer.h
          HINTS "${VC4INFER_ROOT}/include" REQUIRED)
find_library(VC4INFER_LIB NAMES vc4infer
             HINTS "${VC4INFER_ROOT}/build" REQUIRED)

message(STATUS "ggml headers:   ${GGML_INCLUDE_DIR}")
message(STATUS "ggml library:   ${GGML_LIB}")
message(STATUS "vc4infer:       ${VC4INFER_LIB}")

add_library(ggml-vc4-obj OBJECT src/ggml-vc4.cpp)
target_include_directories(ggml-vc4-obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GGML_INCLUDE_DIR}" "${GGML_ROOT}/src" "${VC4INFER_INCLUDE_DIR}")
target_compile_options(ggml-vc4-obj PRIVATE -Wall -Wextra -O2)

# ---------------------------------------------------------------------------
# the dynamic backend: libggml-vc4.so, loaded through GGML_BACKEND_PATH
# ---------------------------------------------------------------------------
add_library(ggml-vc4 SHARED
    $<TARGET_OBJECTS:ggml-vc4-obj>
    src/ggml-vc4-dl.cpp)
target_include_directories(ggml-vc4 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GGML_INCLUDE_DIR}" "${GGML_ROOT}/src")
target_compile_definitions(ggml-vc4 PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED)
target_link_libraries(ggml-vc4 PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}" "${VC4INFER_LIB}")
set_target_properties(ggml-vc4 PROPERTIES
    OUTPUT_NAME "ggml-vc4"
    PREFIX "lib"
    BUILD_RPATH "${VC4INFER_ROOT}/build;${GGML_LIB_DIR}"
    INSTALL_RPATH "${VC4INFER_ROOT}/build;${GGML_LIB_DIR}")

# ---------------------------------------------------------------------------
# the statically registered backend for the ctypes harness (no dlopen)
# ---------------------------------------------------------------------------
add_library(vc4ggml SHARED
    $<TARGET_OBJECTS:ggml-vc4-obj>
    tests/ggml_vc4_tests.cpp)
target_include_directories(vc4ggml PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GGML_INCLUDE_DIR}"
    "${GGML_ROOT}/src" "${VC4INFER_INCLUDE_DIR}")
target_link_libraries(vc4ggml PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}" "${VC4INFER_LIB}")
set_target_properties(vc4ggml PROPERTIES
    BUILD_RPATH "${VC4INFER_ROOT}/build;${GGML_LIB_DIR}")

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
add_executable(test_ggml_vc4
    tests/test_main.cpp
    tests/ggml_vc4_tests.cpp
    $<TARGET_OBJECTS:ggml-vc4-obj>)
target_include_directories(test_ggml_vc4 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GGML_INCLUDE_DIR}"
    "${GGML_ROOT}/src" "${VC4INFER_INCLUDE_DIR}")
target_link_libraries(test_ggml_vc4 PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}" "${GGML_CPU_LIB}" "${VC4INFER_LIB}")
set_target_properties(test_ggml_vc4 PROPERTIES
    BUILD_RPATH "${VC4INFER_ROOT}/build;${GGML_LIB_DIR}")

add_executable(test_dl_load tests/dl_load_main.cpp)
target_include_directories(test_dl_load PRIVATE "${GGML_INCLUDE_DIR}")
target_link_libraries(test_dl_load PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}")
set_target_properties(test_dl_load PROPERTIES
    BUILD_RPATH "${VC4INFER_ROOT}/build;${GGML_LIB_DIR}")

install(TARGETS ggml-vc4 LIBRARY DESTINATION lib)
