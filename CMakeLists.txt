cmake_minimum_required(VERSION 3.19)
project(LBVH LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ext)

set(BENCHMARK_INCLUDE_ALL
    ${FCPW_EIGEN_INCLUDES}
    ${FCPW_ENOKI_INCLUDES}
    ${FCPW_SLANG_INCLUDES}
)

set(BENCHMARK_LIBS_ALL
    fcpw
    ${FCPW_SLANG_LIBRARY}
    ${FCPW_SLANG_GLSLANG_LIBRARY}
    ${FCPW_GFX_LIBRARY}
)

set(EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/example)
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/example)

function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${EXAMPLE_DIR} ${OUTPUT_DIR}
    )
endfunction()

add_cuda_executable(line_test line_test.cu)
add_cuda_executable(triangle_test triangle_test.cu)
add_cuda_executable(benchmark benchmark.cu)

target_compile_definitions(benchmark PRIVATE -DFCPW_USE_GPU)
target_include_directories(benchmark PRIVATE ${BENCHMARK_INCLUDE_ALL})
target_link_libraries(benchmark PRIVATE ${BENCHMARK_LIBS_ALL})
