#pragma once

#include <cuda_runtime.h>

namespace util
{
    /**
     * @brief "Interface" class for object created on CUDA unified memory
     */
    class CudaManaged
    {
    public:
        __host__ void* operator new(size_t len)
        {
            void* ptr;
            cudaMallocManaged(&ptr, len);
            cudaDeviceSynchronize();
            return ptr;
        }

        /**
         * @brief rewrited delete operator for cuda managed object
         * @note cudaDeviceSynchronize in device code is deprecated so using new/delete in
         * device code always causes problem now. But we can still use stack object safely.
         */
        __host__ __device__ void operator delete(void* ptr)
        {
        #ifndef __CUDA_ARCH__
            cudaDeviceSynchronize();
        #endif
            cudaFree(ptr);
        }
    };
}
