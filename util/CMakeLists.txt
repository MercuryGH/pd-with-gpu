cmake_minimum_required(VERSION 3.18)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(util STATIC ${srcs})
target_include_directories(util PUBLIC include)

