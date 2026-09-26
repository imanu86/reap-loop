cmake_minimum_required(VERSION 3.21)
project(qwen36_reap_isolated LANGUAGES C CXX)

# Reuse only the manifest-verified original CUDA12.6/SM86 GGML binaries.
set(REAP_LAB "D:/ds4_work/qwen36_reap_lab")
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${REAP_LAB}/build/bin")
foreach(name ggml-base ggml-cpu ggml-cuda ggml)
    add_library(${name} SHARED IMPORTED GLOBAL)
    set_target_properties(${name} PROPERTIES
        IMPORTED_IMPLIB "${REAP_LAB}/build/verified-libs/${name}.lib"
        IMPORTED_LOCATION "${REAP_LAB}/bin-baseline/${name}.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${REAP_LAB}/source/ggml/include"
        INTERFACE_COMPILE_DEFINITIONS "GGML_SHARED;GGML_BACKEND_SHARED;GGML_USE_CPU;GGML_USE_CUDA")
endforeach()
set_target_properties(ggml PROPERTIES INTERFACE_LINK_LIBRARIES "ggml-base;ggml-cpu;ggml-cuda")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
foreach(option LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_BUILD_MTMD LLAMA_OPENSSL LLAMA_CURL LLAMA_LLGUIDANCE)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_NUMBER 0 CACHE STRING "" FORCE)
set(LLAMA_BUILD_COMMIT "reap-lab-uncommitted" CACHE STRING "" FORCE)
add_subdirectory("${REAP_LAB}/source" llama)

add_executable(reap-native-gate "${CMAKE_CURRENT_LIST_DIR}/../../gate/native_gate.cpp")
target_link_libraries(reap-native-gate PRIVATE llama-common llama)
target_compile_features(reap-native-gate PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(reap-native-gate PRIVATE /utf-8 /bigobj)
endif()
