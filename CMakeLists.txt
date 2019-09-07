cmake_minimum_required (VERSION 2.8.11)
project (HelmDeriv)

find_package(autodiff)

set(CMAKE_CXX_STANDARD 11)
include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/externals/autodiff")
add_executable(main main.cpp)

file(GLOB_RECURSE snippets "${CMAKE_CURRENT_SOURCE_DIR}/snippets/*.cpp")

message(STATUS "snippets found = ${snippets}")
foreach (snippet ${snippets})

	get_filename_component(snippet_name ${snippet} NAME)
	get_filename_component(snippet_exe ${snippet} NAME_WE)
	message(STATUS "snippet_name = ${snippet_name}")

	add_executable        (${snippet_exe} ${snippet})
	target_link_libraries (${snippet_exe} autodiff::autodiff)
endforeach()