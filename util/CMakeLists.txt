cmake_minimum_required(VERSION 3.18)
include(../cmake/utils.cmake)

set(LIB_NAME "util")

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
if (CUDA_ENABLE)
    find_cuda()
    file(GLOB_RECURSE cuda_srcs src/*.cu src/*.cuh)
    if (ALGO_SHARED_LIB)
        message(STATUS "Build util shared library")

        add_library(${LIB_NAME} SHARED ${cuda_srcs} ${srcs})
        set_target_properties(${LIB_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY SHARED)
    else()
        add_library(${LIB_NAME} STATIC ${cuda_srcs} ${srcs})
    endif()
else()
    if (ALGO_SHARED_LIB)
        message(STATUS "Build util shared library")
        add_library(${LIB_NAME} SHARED ${srcs})
    else()
        add_library(${LIB_NAME} STATIC ${srcs})
    endif()
endif()

target_compile_features(${LIB_NAME} PRIVATE cxx_std_20)

target_include_directories(${LIB_NAME} PUBLIC include)

install(
    TARGETS ${LIB_NAME}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    # PUBLIC_HEADER
        # DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
