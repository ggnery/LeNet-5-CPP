cmake_minimum_required(VERSION 3.28.3)
set(PROJECT_NAME "mini_torch")
project(${PROJECT_NAME})

set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)

set(CMAKE_CXX_STANDARD 17)

# Option to enable CUDA (can be set via -DWITH_CUDA=ON)
option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
    add_compile_definitions(TORCH_CUDA_AVAILABLE)
endif()

# Set the path to libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch")

# Find the Torch package
find_package(Torch REQUIRED)

# Project config 
include_directories(src)
include_directories(${TORCH_INCLUDE_DIRS})
file(GLOB_RECURSE PROJECT_FILES src/*.cpp)

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")