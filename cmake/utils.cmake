function(find_cuda)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS ${CUDA_INCLUDE_DIRS})
        include_directories(${CUDA_INCLUDE_DIRS})
    else(CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found")
    endif(CUDA_FOUND)
endfunction()

function(add_headers VAR)
    set(headers ${${VAR}})
    foreach (header ${ARGN})
        set(headers ${headers} ${header})
    endforeach()
    set(${VAR} ${headers} PARENT_SCOPE)
endfunction()
