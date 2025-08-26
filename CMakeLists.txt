cmake_minimum_required(VERSION 3.28.3)
set(PROJECT_NAME "mini_torch")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

include("${CMAKE_CURRENT_SOURCE_DIR}/install/dlib/lib/cmake/dlib/dlibConfig.cmake")

# Project config 
include_directories(src)
file(GLOB_RECURSE PROJECT_FILES src/*.cpp)

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

target_link_libraries(mini_torch dlib::dlib)
