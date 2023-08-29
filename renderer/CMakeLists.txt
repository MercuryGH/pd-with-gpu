cmake_minimum_required(VERSION 3.18)
include(FetchContent)
include(../cmake/utils.cmake)

set(LIBIGL_OPENGL ON  CACHE INTERNAL "libigl with opengl"  )
set(LIBIGL_GLFW   ON  CACHE INTERNAL "libigl with glfw"    )
set(LIBIGL_PNG    ON  CACHE INTERNAL "libigl with png"     )
set(LIBIGL_COPYLEFT_TETGEN   ON  CACHE INTERNAL "libigl with tetgen"  )
set(LIBIGL_IMGUI  ON  CACHE INTERNAL "libigl with imgui"   )

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

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
find_cuda()

add_executable(pd-gui ${srcs})
target_compile_options(pd-gui PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_features(pd-gui PRIVATE cxx_std_20)

target_include_directories(pd-gui PUBLIC include)

find_package(CUDAToolkit REQUIRED) # when linking dynamic lib includes cuda_runtime.h, this is necessary
target_link_libraries(pd-gui
    pd-algo
    CUDA::cudart
    igl::core
    igl::imgui
    # igl::png
    igl_copyleft::tetgen
    util
)

