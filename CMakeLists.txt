cmake_minimum_required (VERSION 2.8.11)
project (HelmDeriv)

set(CMAKE_CXX_STANDARD 11)
include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")
add_executable(main main.cpp)