cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUDA_ENABLE "use cuda" ON)

if (CUDA_ENABLE)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # allow device function in different .cu files
	set(CMAKE_CUDA_ARCHITECTURES 86) # set __CUDA_ARCH__ to be the latest to generate fastest code (require latest device)
	project(projective-dynamics LANGUAGES CXX CUDA)
else()
	project(projective-dynamics LANGUAGES CXX)
endif()

set_property(GLOBAL PROPERTY USE_FOLDER ON)

add_subdirectory(pd)
add_subdirectory(util)

