#include <pd/cutest.h>
#include <cstdio>

__host__ __device__ void pd::say_hello() {
#ifdef __CUDA_ARCH__
	printf("Hello, world from GPU architecture %d!\n", __CUDA_ARCH__);
#else
	printf("Hello, world from CPU!\n");
#endif
}

__global__ void pd::kernel() {
	say_hello();
}

int pd::test()
{
	kernel<<<1, 1>>>();
	cudaDeviceSynchronize();
	say_hello();
	return 0;
}

