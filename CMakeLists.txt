cmake_minimum_required (VERSION 2.8.11)
project (HelmDeriv)

find_package(autodiff)

set(CMAKE_CXX_STANDARD 11)
include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/externals/autodiff")
add_executable(main main.cpp)

add_executable(eigen_eos eigen_eos.cpp)
target_link_libraries(eigen_eos autodiff::autodiff)