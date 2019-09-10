cmake_minimum_required (VERSION 2.8.11)
project (HelmDeriv)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)

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

	set_target_properties(${snippet_exe} PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
        )
endforeach()