cmake_minimum_required(VERSION 3.18)
include(FetchContent)

set(LIBIGL_OPENGL ON  CACHE INTERNAL "libigl with opengl"  )
set(LIBIGL_GLFW   ON  CACHE INTERNAL "libigl with glfw"    )
set(LIBIGL_IMGUI  ON  CACHE INTERNAL "libigl with imgui"   )

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  _libigl
  GIT_REPOSITORY https://github.com/libigl/libigl
  GIT_TAG        v2.4.0
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(_libigl)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
if (CUDA_ENABLE)
    include_directories("E:\\CUDA\\v11.6\\include")
    file(GLOB_RECURSE cuda_srcs src/*.cu)
    add_executable(pd ${cuda_srcs} ${srcs})
else()
    add_executable(pd ${srcs})
endif()

#add_executable(pd ${srcs})
target_compile_features(pd PRIVATE cxx_std_17)

target_include_directories(pd PUBLIC include)

target_link_libraries(pd 
PRIVATE 
    igl::core 
    igl::imgui
)


