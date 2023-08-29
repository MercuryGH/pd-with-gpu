cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Installing the library
set(CMAKE_INSTALL_PREFIX /home/xinghai/codes/pd-with-gpu/build/install)
set(CMAKE_INSTALL_LIBDIR ./artifact)
set(CMAKE_INSTALL_INCLUDEDIR ./include)

option(CUDA_ENABLE "use cuda" ON)
message(STATUS "CUDA enable: ${CUDA_ENABLE}")
option(ALGO_SHARED_LIB "build pd-algo shared (dynamic-linked at bootup) lib" ON)
message(STATUS "Build shared library: ${ALGO_SHARED_LIB}")

if (CUDA_ENABLE)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # allow device function in different .cu files
	set(CMAKE_CUDA_ARCHITECTURES 86) # set __CUDA_ARCH__ to be the latest to generate fastest code (require latest device)
	project(projective-dynamics LANGUAGES CXX CUDA)
else()
	project(projective-dynamics LANGUAGES CXX)
endif()

set_property(GLOBAL PROPERTY USE_FOLDER ON)

add_subdirectory(util)
add_subdirectory(pd)
add_subdirectory(renderer)

