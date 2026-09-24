cmake_minimum_required(VERSION 3.21)

project(ps2xTest LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(gs_cache)
endif()

add_library(ps2_gs_reference STATIC
    gs_reference/gs_reference_backend.cpp
    gs_reference/gs_golden.cpp
)

target_include_directories(ps2_gs_reference PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/gs_reference
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/include
)

target_link_libraries(ps2_gs_reference PUBLIC ps2_runtime)

if(MSVC)
    set_source_files_properties(gs_reference/gs_reference_backend.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
endif()

add_executable(gs_replay tools/gs_replay.cpp)
target_link_libraries(gs_replay PRIVATE ps2_gs_reference)
if(MSVC)
    target_compile_options(gs_replay PRIVATE /Zi)
    target_link_options(gs_replay PRIVATE /DEBUG)
endif()

# Static library with test logic (no main), used by ps2xStudio
add_library(ps2_test_function_table OBJECT
    src/test_function_table.cpp
)

target_include_directories(ps2_test_function_table PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xIOP/include
)

add_library(ps2_test_lib STATIC
    src/code_generator_tests.cpp
    src/r5900_decoder_tests.cpp
    src/elf_analyzer_tests.cpp
    src/pad_input_tests.cpp
    src/ps2_runtime_io_tests.cpp
    src/ps2_runtime_kernel_tests.cpp
    src/ps2_runtime_interrupt_tests.cpp
    src/ps2_memory_tests.cpp
    src/ps2_vu1_tests.cpp
    src/ps2_vu_tests.cpp
    src/ps2_gs_tests.cpp
    src/ps2_iop_tests.cpp
    src/ps2_sif_rpc_tests.cpp
    src/ps2_sif_dma_tests.cpp
    src/ps2_recompiler_tests.cpp
    src/ps2_runtime_expansion_tests.cpp
    src/gs_golden_tests.cpp
)

target_include_directories(ps2_test_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRecomp/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xAnalyzer/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/src/lib
)

target_link_libraries(ps2_test_lib PRIVATE
    ps2_gs_reference
    ps2_recomp_lib
    ps2_analyzer_lib
    ps2_runtime
    ps2_iop
)

if(UNIX AND NOT APPLE)
    target_link_libraries(ps2_test_lib PRIVATE ${CMAKE_DL_LIBS})
endif()

add_executable(ps2x_tests
    src/main.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

option(PRINT_GENERATED_CODE "Print generated code in tests" OFF)
if(PRINT_GENERATED_CODE)

    target_compile_definitions(ps2x_tests PRIVATE PRINT_GENERATED_CODE)
    target_compile_definitions(ps2_test_lib PRIVATE PRINT_GENERATED_CODE)
endif()

target_include_directories(ps2x_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRecomp/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xAnalyzer/include
    ${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/include
)

target_link_libraries(ps2x_tests PRIVATE
    ps2_test_lib
    ps2_recomp_lib
    ps2_analyzer_lib
    ps2_runtime
)

# VU/GS tests intentionally instantiate large interpreter state.
# Windows executables default to a much smaller stack than Linux, and
# sanitizer/debug instrumentation can push these test frames over 1 MiB.
if(MSVC)
    target_link_options(ps2x_tests PRIVATE "/STACK:8388608")
elseif(MINGW)
    target_link_options(ps2x_tests PRIVATE "--stack,8388608")
endif()

if(COMMAND ps2x_stage_ffmpeg_runtime_dlls)
    ps2x_stage_ffmpeg_runtime_dlls(ps2x_tests)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../ps2xRuntime/cmake/ReleaseMode.cmake")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    EnableFastReleaseMode(ps2_test_lib)
    EnableFastReleaseMode(ps2x_tests)
    EnableFastReleaseMode(ps2_gs_reference)
    EnableFastReleaseMode(gs_replay)
endif()
