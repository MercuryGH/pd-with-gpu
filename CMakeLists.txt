cmake_minimum_required(VERSION 3.18)

option(CUDA_ENABLE "use cuda" ON)

if (CUDA_ENABLE)
	project(projective-dynamics LANGUAGES CXX CUDA)
else()
	project(projective-dynamics LANGUAGES CXX)
endif()

set_property(GLOBAL PROPERTY USE_FOLDER ON)

add_subdirectory(pd)
add_subdirectory(util)