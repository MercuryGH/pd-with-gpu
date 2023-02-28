cmake_minimum_required(VERSION 3.18)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
if (CUDA_ENABLE)
    include_directories("/opt/cuda/include")
    file(GLOB_RECURSE cuda_srcs src/*.cu)
    add_library(util STATIC ${cuda_srcs} ${srcs})
else()
    add_library(util STATIC ${srcs})
endif()

target_compile_features(util PRIVATE cxx_std_17)

target_include_directories(util PUBLIC include)

