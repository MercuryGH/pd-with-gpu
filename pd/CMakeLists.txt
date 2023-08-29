cmake_minimum_required(VERSION 3.18)
include(FetchContent)
include(../cmake/utils.cmake)

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  _libigl
  GIT_REPOSITORY https://github.com/libigl/libigl
#   GIT_TAG        v2.4.0
# use the latest version is OK
  GIT_TAG        main
  # GIT_TAG      5ded7da0868123b75969ebe548cb71abedf5cec9
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(_libigl)

if (CUDA_ENABLE)
    set(LIB_NAME "pd-algo")

    file(GLOB_RECURSE algo_headers CONFIGURE_DEPENDS include/*.h)
    file(GLOB_RECURSE algo_srcs CONFIGURE_DEPENDS src/*.cpp src/*.cu)
    add_headers(export_headers
        include/io/io_data.h
        include/io/mesh_io.h
        include/io/simulation_control.h

        include/pd/deformable_mesh.h
        include/pd/types.h
        include/pd/algo_ctrl.h

        include/primitive/primitive.h

        ../util/include/util/singleton.h
    )
    # file(GLOB_RECURSE export_headers CONFIGURE_DEPENDS include/*.h )

    message(STATUS "header is ${export_headers}")
    find_cuda()

    add_library(${LIB_NAME} SHARED ${algo_headers} ${algo_srcs})
    target_compile_options(${LIB_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_features(${LIB_NAME} PUBLIC cxx_std_20)
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        PUBLIC_HEADER "${export_headers}" # if don't use quotes, only the 1st header is installed
    )
    target_include_directories(${LIB_NAME} PUBLIC include)

    target_link_libraries(${LIB_NAME}
        PUBLIC
        igl::core
        util
    )

    install(
        TARGETS ${LIB_NAME}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # install(
        # FILES ${CMAKE_BINARY_DIR}/mylib.pc
        # DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig
    # )

endif()
