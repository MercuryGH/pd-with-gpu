cmake_minimum_required(VERSION 3.18)
include(FetchContent)

set(LIBIGL_OPENGL ON  CACHE INTERNAL "libigl with opengl"  )
set(LIBIGL_GLFW   ON  CACHE INTERNAL "libigl with glfw"    )
set(LIBIGL_PNG    ON  CACHE INTERNAL "libigl with png"     )
set(LIBIGL_COPYLEFT_TETGEN   ON  CACHE INTERNAL "libigl with tetgen"  )
set(LIBIGL_IMGUI  ON  CACHE INTERNAL "libigl with imgui"   )

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  _libigl
  GIT_REPOSITORY https://github.com/libigl/libigl
  GIT_TAG        v2.4.0
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(_libigl)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # allow device function in different .cu files
set(CMAKE_CUDA_ARCHITECTURES 86) # set __CUDA_ARCH__ to be the latest to generate fastest code (require latest device)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
if (CUDA_ENABLE)
    include_directories("/opt/cuda/include")
    file(GLOB_RECURSE cuda_srcs src/*.cu)
    add_executable(pd ${cuda_srcs} ${srcs})
else()
    add_executable(pd ${srcs})
endif()

#add_executable(pd ${srcs})
target_compile_options(pd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_features(pd PRIVATE cxx_std_17)

target_include_directories(pd PUBLIC include)

target_link_libraries(pd 
PRIVATE 
    igl::core 
    igl::imgui
    igl::png
    igl_copyleft::tetgen
    util
)

