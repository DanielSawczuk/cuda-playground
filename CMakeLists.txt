cmake_minimum_required(VERSION 3.22.1)
project(CUDA-Playground LANGUAGES CUDA CXX)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)
