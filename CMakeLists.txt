cmake_minimum_required(VERSION 3.28.3)
project(mini_torch)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

include("${CMAKE_CURRENT_SOURCE_DIR}/install/dlib/lib/cmake/dlib/dlibConfig.cmake")
add_executable(mini_torch main.cpp)

target_link_libraries(mini_torch dlib::dlib)
