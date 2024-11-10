set(TARGET gemm_example)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} cuda)
add_subdirectory(kernels)
