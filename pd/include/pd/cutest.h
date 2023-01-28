#pragma once
#include <cuda_runtime.h>

namespace pd
{
	__host__ __device__ void say_hello();
	__global__ void kernel();
	int test();
}