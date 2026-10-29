# Courier's engine: speech detection and the wake word (onnxruntime), recognition (whisper.cpp), the
# language model (llama.cpp) and the voice (Piper on onnxruntime), as one shared library with a C API
# (include/courier.h) for the Rust orchestrator and the Python wake word training.
#
#   cmake -S engine -B engine/build && cmake --build engine/build --config Release
#   with an NVIDIA GPU (CUDA), from a "x64 Native Tools" prompt on Windows:
#   cmake -S engine -B engine/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOURIER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake_minimum_required(VERSION 3.22)
project(courier_engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(COURIER_CUDA "Run Whisper and the language model on an NVIDIA GPU" OFF)
option(COURIER_TESTS "Build the tests" ON)
set(COURIER_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party" CACHE PATH "Where scripts/fetch-deps.sh put things")
get_filename_component(COURIER_THIRD_PARTY "${COURIER_THIRD_PARTY}" ABSOLUTE)

if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")  # not for nvcc
endif()

# llama.cpp first: it builds ggml, which whisper.cpp then reuses (one copy of ggml).
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${COURIER_CUDA} CACHE BOOL "" FORCE)
add_subdirectory("${COURIER_THIRD_PARTY}/llama.cpp" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${COURIER_THIRD_PARTY}/whisper.cpp" "${CMAKE_BINARY_DIR}/whisper.cpp" EXCLUDE_FROM_ALL)

# onnxruntime, prebuilt (scripts/fetch-deps.sh).
set(ORT_DIR "${COURIER_THIRD_PARTY}/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DIR}/lib/onnxruntime.dll" IMPORTED_IMPLIB "${ORT_DIR}/lib/onnxruntime.lib")
else()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.so")
endif()
set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include")

# The parts, as a static library: the shared library and the tests link it.
add_library(courier_core STATIC src/onnx.cpp src/vad.cpp src/wake.cpp src/asr.cpp src/llm.cpp src/g2p.cpp src/tts.cpp)
target_include_directories(courier_core PUBLIC src include "${COURIER_THIRD_PARTY}/llama.cpp/vendor")
target_link_libraries(courier_core PUBLIC whisper llama llama-common onnxruntime)

add_library(courier_engine SHARED src/api.cpp)
target_link_libraries(courier_engine PRIVATE courier_core)
target_compile_definitions(courier_engine PRIVATE COURIER_BUILDING)
if(COURIER_CUDA)
  target_compile_definitions(courier_engine PRIVATE COURIER_CUDA)
endif()
if(NOT MSVC)
  target_compile_options(courier_engine PRIVATE -fvisibility=hidden)
  # Find onnxruntime next to the library.
  set_target_properties(courier_engine PROPERTIES BUILD_RPATH "$ORIGIN;${ORT_DIR}/lib" INSTALL_RPATH "$ORIGIN")
endif()
# onnxruntime next to the library, so one folder holds everything the orchestrator loads.
add_custom_command(TARGET courier_engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime> $<TARGET_FILE_DIR:courier_engine>)

if(COURIER_TESTS)
  enable_testing()
  add_executable(engine_tests tests/engine_tests.cpp)
  target_link_libraries(engine_tests PRIVATE courier_core)
  target_compile_definitions(engine_tests PRIVATE COURIER_MODELS="${COURIER_THIRD_PARTY}/models"
                             COURIER_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME engine_tests COMMAND engine_tests)
  add_dependencies(engine_tests courier_engine)  # for the onnxruntime copy
endif()
